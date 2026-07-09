import random
from dataclasses import dataclass, field
from skill_queue import SkillQueue


@dataclass
class Combatant:
    """
    Shared state between PlayerUnit and Boss: HP, stagger, and all status
    effects (tremor, burn, dark flame, magic bullets). PlayerUnit and Boss
    each just add what's specific to them (skills, and for PlayerUnit, its
    own skill queue).
    """
    name: str
    max_hp: int
    passive_description: str = ""
    stagger_thresholds: list = field(default_factory=list)  # absolute HP values, e.g. [110, 70, 30]
    hp: int = field(init=False)
    is_staggered: bool = field(init=False, default=False)
    stagger_turns_left: int = field(init=False, default=0)
    passed_thresholds: set = field(init=False, default_factory=set)
    tremor_potency: int = 0
    tremor_count: int = 0
    tremor_type: str = "normal"  # "normal" or "scorch"
    tremor_scorch_expire_turn: object = None  # int turn number, or None
    burn_potency: int = 0
    burn_count: int = 0
    burn_nonlethal: bool = False  # True if current burn stack can't reduce HP below 1
    dark_flame_count: int = 0
    magic_bullets: int = 0
    max_magic_bullets: int = 0  # 0 = doesn't use this mechanic

    def __post_init__(self):
        self.hp = self.max_hp

    def is_alive(self):
        return self.hp > 0

    def active_stagger_thresholds(self):
        """Thresholds not yet passed (still meaningful to show on the HP bar)."""
        return [t for t in self.stagger_thresholds if t not in self.passed_thresholds]


@dataclass
class SkillDef:
    name: str
    roll_lo: int
    roll_hi: int
    base_damage: int
    description: str = ""  # full effect text, including not-yet-implemented status effects
    effect: object = None  # optional callable(battle, source, target, log); None -> flat damage only


@dataclass
class PlayerUnit(Combatant):
    skill1: SkillDef = None
    skill2: SkillDef = None
    queue: SkillQueue = field(init=False, default=None)

    def init_queue(self, rng):
        self.queue = SkillQueue(rng)

    def get_skill_def(self, skill_type):
        return self.skill1 if skill_type == 'skill1' else self.skill2


@dataclass
class BossSkillDef:
    name: str
    roll_lo: int
    roll_hi: int
    base_damage: int
    hits_all: bool = False
    description: str = ""
    effect: object = None  # optional callable(battle, source, target, log); None -> flat damage only


@dataclass
class Boss(Combatant):
    skills: list = None  # list of 5 BossSkillDef


class PlannedBossSkill:
    """One of the boss's 3 chosen skills for this turn, plus its target(s)."""
    def __init__(self, skill_def, target_names):
        self.skill_def = skill_def
        self.target_names = target_names  # list of unit names (usually 1, or all for hits_all)
        self.discarded = False

    def __repr__(self):
        return f"<BossSlot {self.skill_def.name} -> {self.target_names}>"


class PlayerAction:
    """A player's chosen skill for one of their units this turn."""
    def __init__(self, unit_name, skill_type, clash_slot_index=None, armed_position=None):
        self.unit_name = unit_name
        self.skill_type = skill_type
        self.clash_slot_index = clash_slot_index  # None = unopposed, else index into boss slots
        # armed_position (0 or 1): which of the two available slots this came from.
        # Only needed to disambiguate when both available slots share the same skill_type;
        # otherwise safe to leave as None.
        self.armed_position = armed_position
        self.discarded = False

    def __repr__(self):
        mode = f"clash->slot{self.clash_slot_index}" if self.clash_slot_index is not None else "unopposed"
        return f"<PlayerAction {self.unit_name}:{self.skill_type} {mode}>"


class TurnStep:
    """
    One atomic, animatable piece of a turn's resolution: a single boss slot
    (whether clashed or unopposed), a single unopposed player action, or the
    end-of-turn status effect processing. A UI can iterate these one at a
    time (via Battle.resolve_turn_steps) to animate a turn instead of
    resolving it all at once.
    """
    def __init__(self, kind, participants, log_lines=None, slot_index=None,
                 boss_roll=None, player_roll=None, boss_skill_name=None,
                 player_skill_name=None, winner=None):
        self.kind = kind  # 'clash' | 'boss_unopposed' | 'player_unopposed' | 'turn_end'
        self.participants = participants  # entity names involved, for UI highlighting
        self.log_lines = log_lines or []
        self.slot_index = slot_index  # which boss slot (0-2), if applicable
        self.boss_roll = boss_roll
        self.player_roll = player_roll
        self.boss_skill_name = boss_skill_name
        self.player_skill_name = player_skill_name
        self.winner = winner  # 'boss' | 'player', only set for 'clash'

    def __repr__(self):
        return f"<TurnStep {self.kind} participants={self.participants}>"


class BattleLog:
    def __init__(self):
        self.lines = []

    def add(self, line):
        self.lines.append(line)

    def dump(self):
        return "\n".join(self.lines)


class Battle:
    def __init__(self, boss, units, rng=None):
        self.boss = boss
        self.units = {u.name: u for u in units}
        self.rng = rng or random.Random()
        for u in units:
            u.init_queue(self.rng)
        self.turn_number = 0

    def get_unit(self, name):
        return self.units[name]

    def alive_units(self):
        return [u for u in self.units.values() if u.is_alive()]

    def boss_choose_turn(self):
        """Boss picks 3 distinct random skills from its 5, and random target(s) for each."""
        chosen = self.rng.sample(self.boss.skills, 3)
        alive_names = [u.name for u in self.alive_units()]
        slots = []
        for skill_def in chosen:
            if skill_def.hits_all:
                targets = list(alive_names)
            else:
                targets = [self.rng.choice(alive_names)]
            slots.append(PlannedBossSkill(skill_def, targets))
        return slots

    def _roll(self, lo, hi):
        return self.rng.randint(lo, hi)

    def resolve_turn(self, boss_slots, player_actions):
        """
        boss_slots: list of PlannedBossSkill (length 3), in chosen order (leftmost first).
        player_actions: list of PlayerAction, one per living unit that is taking an action
                         this turn (should be exactly one skill per unit, since the bottom
                         2 available slots always offer exactly one action to pick per unit
                         per turn).
        Returns a BattleLog with the whole turn resolved at once. For a step-by-step
        (animatable) version, use resolve_turn_steps instead — both share the same
        underlying resolution logic.
        """
        log = BattleLog()
        for _step in self._resolve_turn_impl(boss_slots, player_actions, log):
            pass  # steps already write into `log` as they go; just drain the generator
        return log

    def resolve_turn_steps(self, boss_slots, player_actions):
        """
        Generator version of resolve_turn: yields a TurnStep after each atomic
        action (one boss slot's resolution, one unopposed player action, or the
        end-of-turn status processing), with all state changes for that step
        already applied by the time it's yielded. A UI can pace itself between
        yields (e.g. wait ~1s) to animate the turn instead of resolving it all
        at once.
        """
        log = BattleLog()
        yield from self._resolve_turn_impl(boss_slots, player_actions, log)

    def _resolve_turn_impl(self, boss_slots, player_actions, log):
        self.turn_number += 1
        log.add(f"=== Turn {self.turn_number} ===")

        # Wire up clashes: for quick lookup, map clash_slot_index -> PlayerAction
        clash_map = {}
        for action in player_actions:
            if action.clash_slot_index is not None:
                clash_map[action.clash_slot_index] = action

        # Step 1: boss skills, leftmost first
        for i, slot in enumerate(boss_slots):
            clashing_action = clash_map.get(i)
            start_idx = len(log.lines)
            if clashing_action is not None:
                boss_roll, player_roll, winner, boss_skill_name, player_skill_name = \
                    self._resolve_clash(slot, clashing_action, log)
                yield TurnStep(
                    kind="clash",
                    participants=[self.boss.name, clashing_action.unit_name],
                    log_lines=log.lines[start_idx:],
                    slot_index=i,
                    boss_roll=boss_roll, player_roll=player_roll, winner=winner,
                    boss_skill_name=boss_skill_name, player_skill_name=player_skill_name,
                )
            else:
                self._execute_boss_skill(slot, log)
                yield TurnStep(
                    kind="boss_unopposed",
                    participants=[self.boss.name] + list(slot.target_names),
                    log_lines=log.lines[start_idx:],
                    slot_index=i,
                    boss_skill_name=slot.skill_def.name,
                )

        # Step 2: remaining (unopposed) player actions, in deployment order
        # (deployment order = the order units were passed into Battle(), i.e. dict insertion order)
        for action in player_actions:
            if action.discarded:
                continue
            if action.clash_slot_index is not None:
                continue
            start_idx = len(log.lines)
            self._execute_player_skill(action, log)
            yield TurnStep(
                kind="player_unopposed",
                participants=[action.unit_name, self.boss.name],
                log_lines=log.lines[start_idx:],
                player_skill_name=action.skill_type,
            )

        start_idx = len(log.lines)
        self._process_status_effects_turn_end(log)
        self._process_stagger_turn_end(log)
        yield TurnStep(
            kind="turn_end",
            participants=[],
            log_lines=log.lines[start_idx:],
        )

    def _apply_damage(self, target, base_amount, log, floor_at_1=False):
        """
        Applies damage to target (a PlayerUnit or Boss), accounting for the
        1.5x multiplier while staggered, then checks whether this damage
        crosses a new stagger threshold. If floor_at_1 is True, this hit
        cannot reduce HP below 1 (used for the self-status unit's own burn).
        """
        was_staggered = target.is_staggered
        dmg = int(round(base_amount * 1.5)) if was_staggered else base_amount
        if floor_at_1:
            dmg = min(dmg, max(0, target.hp - 1))
        target.hp -= dmg
        if was_staggered:
            log.add(f"  ({target.name} is staggered: damage x1.5 -> {dmg})")
        self._check_stagger(target, log)
        return dmg

    def _check_stagger(self, target, log):
        """
        Checks target's stagger thresholds against its current HP. Newly-passed
        thresholds are removed permanently. The first newly-passed threshold
        while not already staggered triggers a fresh 2-turn stagger; if the
        target is already staggered, passing further thresholds just removes
        them with no additional effect.
        """
        for threshold in target.stagger_thresholds:
            if threshold in target.passed_thresholds:
                continue
            if target.hp < threshold:
                target.passed_thresholds.add(threshold)
                if not target.is_staggered:
                    target.is_staggered = True
                    target.stagger_turns_left = 1
                    log.add(f"  {target.name} is STAGGERED! (passed threshold {threshold})")
                else:
                    log.add(f"  {target.name} passes another stagger threshold ({threshold}); already staggered.")

    def _process_stagger_turn_end(self, log):
        """Ticks down stagger duration for everyone at the end of a turn's resolution."""
        entities = list(self.units.values()) + [self.boss]
        for entity in entities:
            if not entity.is_staggered:
                continue
            if entity.stagger_turns_left > 0:
                entity.stagger_turns_left -= 1
            else:
                entity.is_staggered = False
                log.add(f"{entity.name} is no longer staggered.")

    # ---------------- burn / tremor / dark flame / magic bullets ----------------

    def _apply_tremor(self, target, potency, count, log):
        target.tremor_potency += potency
        target.tremor_count += count
        if potency or count:
            log.add(f"  {target.name} gains {potency} tremor potency, {count} tremor count "
                     f"(now {target.tremor_potency}/{target.tremor_count}).")

    def _apply_burn(self, target, potency, count, log, nonlethal=False):
        target.burn_potency += potency
        target.burn_count += count
        target.burn_nonlethal = nonlethal
        if potency or count:
            log.add(f"  {target.name} gains {potency} burn potency, {count} burn count "
                     f"(now {target.burn_potency}/{target.burn_count}).")

    def _apply_dark_flame(self, target, count, log):
        target.dark_flame_count += count
        if count:
            log.add(f"  {target.name} gains {count} dark flame (now {target.dark_flame_count}).")

    def _trigger_amplitude_conversion(self, target, log):
        target.tremor_type = "scorch"
        target.tremor_scorch_expire_turn = self.turn_number + 2
        log.add(f"  {target.name}'s tremor converts to TREMOR SCORCH (amplitude conversion).")

    def _tremor_burst(self, target, log):
        """
        Raises all of target's not-yet-passed stagger thresholds by its current
        tremor potency (staggering it if HP now falls below one), reduces tremor
        count by 1, and — if the tremor type is currently 'scorch' — also deals
        physical damage equal to tremor potency + burn potency and ticks burn
        count down by 1. If tremor count reaches 0 as a result, tremor potency
        is cleared too (an expired tremor stack shouldn't linger with stale
        potency still sitting on the unit).
        """
        burst_potency = target.tremor_potency  # snapshot before any reset below

        for idx, t in enumerate(target.stagger_thresholds):
            if t not in target.passed_thresholds:
                target.stagger_thresholds[idx] = t + burst_potency
        log.add(f"  {target.name}'s tremor bursts! (potency {burst_potency}) "
                 f"stagger thresholds raised.")
        self._check_stagger(target, log)
        target.tremor_count = max(0, target.tremor_count - 1)

        if target.tremor_type == "scorch":
            scorch_dmg_base = burst_potency + target.burn_potency
            dmg = self._apply_damage(target, scorch_dmg_base, log)
            log.add(f"  {target.name} takes {dmg} tremor SCORCH damage "
                     f"(tremor potency + burn potency).")
            if target.burn_count > 0:
                target.burn_count -= 1

        if target.tremor_count == 0:
            target.tremor_potency = 0

    def _process_status_effects_turn_end(self, log):
        """
        End-of-turn burn/dark-flame damage, tremor count decay, and amplitude
        conversion expiry. Order relative to the stagger duration tick doesn't
        matter functionally.
        """
        entities = list(self.units.values()) + [self.boss]
        for entity in entities:
            if entity.burn_potency > 0:
                if entity.dark_flame_count > 0:
                    base = entity.dark_flame_count * entity.burn_potency
                    dmg = self._apply_damage(entity, base, log, floor_at_1=entity.burn_nonlethal)
                    log.add(f"{entity.name} takes {dmg} dark flame burn damage "
                             f"({entity.dark_flame_count} dark flame x {entity.burn_potency} burn potency).")
                    entity.dark_flame_count = 0
                else:
                    dmg = self._apply_damage(entity, entity.burn_potency, log, floor_at_1=entity.burn_nonlethal)
                    log.add(f"{entity.name} takes {dmg} burn damage (potency {entity.burn_potency}).")

            if entity.burn_count > 0:
                entity.burn_count -= 1
                if entity.burn_count == 0:
                    entity.burn_potency = 0

            if entity.tremor_count > 0:
                entity.tremor_count -= 1
                if entity.tremor_count == 0:
                    entity.tremor_potency = 0

            if entity.tremor_type == "scorch" and entity.tremor_scorch_expire_turn is not None:
                if self.turn_number >= entity.tremor_scorch_expire_turn:
                    entity.tremor_type = "normal"
                    entity.tremor_scorch_expire_turn = None
                    log.add(f"{entity.name}'s tremor scorch reverts to normal tremor.")

    def _resolve_clash(self, boss_slot, player_action, log):
        boss_def = boss_slot.skill_def
        unit = self.get_unit(player_action.unit_name)
        player_def = unit.get_skill_def(player_action.skill_type)

        while True:
            boss_roll = self._roll(boss_def.roll_lo, boss_def.roll_hi)
            player_roll = self._roll(player_def.roll_lo, player_def.roll_hi)
            log.add(
                f"CLASH: {boss_def.name} rolled {boss_roll} "
                f"vs {unit.name}'s {player_def.name} rolled {player_roll}"
            )
            if boss_roll == player_roll:
                log.add("  Tie! Rerolling both.")
                continue
            break

        if boss_roll > player_roll:
            log.add(f"  Boss wins clash. {player_def.name} is discarded.")
            player_action.discarded = True
            if not boss_def.hits_all:
                # The boss's attack follows through on whoever clashed it, regardless
                # of who it was originally, randomly aimed at.
                if boss_slot.target_names != [unit.name]:
                    log.add(f"  {boss_def.name} redirects onto {unit.name} (the clasher).")
                boss_slot.target_names = [unit.name]
            self._execute_boss_skill(boss_slot, log)
            winner = "boss"
        else:
            log.add(f"  {unit.name} wins clash. {boss_def.name} is discarded.")
            boss_slot.discarded = True
            self._execute_player_skill(player_action, log)
            winner = "player"

        return boss_roll, player_roll, winner, boss_def.name, player_def.name

    def _execute_boss_skill(self, slot, log):
        if slot.discarded:
            return
        if self.boss.is_staggered:
            log.add(f"BOSS is staggered and cannot use {slot.skill_def.name} this slot.")
            return
        skill_def = slot.skill_def
        for target_name in slot.target_names:
            target = self.get_unit(target_name)
            if not target.is_alive():
                continue
            if skill_def.effect is not None:
                skill_def.effect(self, self.boss, target, log)
            else:
                dmg = self._apply_damage(target, skill_def.base_damage, log)
                log.add(
                    f"BOSS uses {skill_def.name} on {target_name}: "
                    f"{dmg} damage (hp now {target.hp})"
                )

    def _execute_player_skill(self, action, log):
        if action.discarded:
            return
        unit = self.get_unit(action.unit_name)
        if unit.is_staggered:
            log.add(f"{unit.name} is staggered and cannot act; its skill is saved for next turn.")
            return
        skill_def = unit.get_skill_def(action.skill_type)
        if skill_def.effect is not None:
            skill_def.effect(self, unit, self.boss, log)
        else:
            dmg = self._apply_damage(self.boss, skill_def.base_damage, log)
            log.add(
                f"{unit.name} uses {skill_def.name}: "
                f"{dmg} damage to {self.boss.name} (boss hp now {self.boss.hp})"
            )
        # advance this unit's skill queue now that the skill has actually been used
        unit.queue.pick(action.skill_type, position=action.armed_position)


# ----------------------------------------------------------------------------
# SKILL EFFECT LIBRARY
#
# Each function has signature (battle, source, target, log) and is attached to
# a SkillDef/BossSkillDef via its `effect=` field. `source` is the acting unit
# (or the Boss); `target` is whoever the skill is aimed at (the Boss, for all
# player skills; a specific party unit, for boss skills).
#
# NOTE: BossSkillDef "Clash Baiter"'s clash-triggered +5 roll / +900% damage
# bonus is a roll/clash-time mechanic, not a target-effect, and is NOT yet
# implemented here — it still just deals its flat base damage.
# ----------------------------------------------------------------------------

def effect_tremor_scorch_skill1(battle, source, target, log):
    dmg = battle._apply_damage(target, 25, log)
    log.add(f"{source.name} uses Tremor Jab on {target.name}: {dmg} damage (hp now {target.hp}).")
    battle._apply_tremor(target, potency=3, count=2, log=log)
    if target.tremor_count > 3:
        battle._tremor_burst(target, log)


def effect_tremor_scorch_skill2(battle, source, target, log):
    dmg = battle._apply_damage(target, 50, log)
    log.add(f"{source.name} uses Tremor Burst Strike on {target.name}: {dmg} damage (hp now {target.hp}).")
    battle._apply_tremor(target, potency=2, count=0, log=log)
    # Convert to scorch BEFORE bursting, so both bursts (including this one) count as scorch.
    battle._trigger_amplitude_conversion(target, log)
    battle._tremor_burst(target, log)
    battle._tremor_burst(target, log)


def effect_dark_flame_skill1(battle, source, target, log):
    dmg = battle._apply_damage(target, 25, log)
    log.add(f"{source.name} uses Flame Tag on {target.name}: {dmg} damage (hp now {target.hp}).")
    battle._apply_burn(target, potency=source.magic_bullets, count=0, log=log)
    battle._apply_dark_flame(target, count=1, log=log)
    source.magic_bullets = min(source.magic_bullets + 2, source.max_magic_bullets)
    log.add(f"  {source.name} gains 2 magic bullets (now {source.magic_bullets}/{source.max_magic_bullets}).")


def effect_dark_flame_skill2(battle, source, target, log):
    dmg = battle._apply_damage(target, 50, log)
    log.add(f"{source.name} uses Dark Flame Surge on {target.name}: {dmg} damage (hp now {target.hp}).")
    source.magic_bullets = min(source.magic_bullets + 1, source.max_magic_bullets)
    log.add(f"  {source.name} gains 1 magic bullet (now {source.magic_bullets}/{source.max_magic_bullets}).")
    battle._apply_dark_flame(target, count=source.magic_bullets, log=log)


def effect_self_status_skill1(battle, source, target, log):
    dmg = battle._apply_damage(target, 25, log)
    log.add(f"{source.name} uses Shared Tremor on {target.name}: {dmg} damage (hp now {target.hp}).")
    battle._apply_tremor(target, potency=16, count=8, log=log)
    battle._apply_tremor(source, potency=5, count=1, log=log)


def effect_self_status_skill2(battle, source, target, log):
    base = 50 + source.burn_potency
    dmg = battle._apply_damage(target, base, log)
    log.add(f"{source.name} uses Shared Burn on {target.name}: {dmg} damage (hp now {target.hp}).")
    battle._apply_burn(target, potency=10, count=5, log=log, nonlethal=True)
    battle._apply_burn(source, potency=10, count=5, log=log, nonlethal=True)


def effect_boss_tremor_slam(battle, source, target, log):
    dmg = battle._apply_damage(target, 25, log)
    log.add(f"BOSS uses Tremor Slam on {target.name}: {dmg} damage (hp now {target.hp}).")
    battle._apply_tremor(target, potency=5, count=3, log=log)
    battle._tremor_burst(target, log)


def effect_boss_burn_wave(battle, source, target, log):
    dmg = battle._apply_damage(target, 10, log)
    log.add(f"BOSS uses Burn Wave on {target.name}: {dmg} damage (hp now {target.hp}).")
    battle._apply_burn(target, potency=10, count=3, log=log)


def effect_boss_clash_baiter(battle, source, target, log):
    dmg = battle._apply_damage(target, 15, log)
    log.add(f"BOSS uses Clash Baiter on {target.name}: {dmg} damage (hp now {target.hp}).")
    battle._apply_tremor(target, potency=3, count=0, log=log)


def effect_boss_scorch_point(battle, source, target, log):
    dmg = battle._apply_damage(target, 20, log)
    log.add(f"BOSS uses Scorch Point on {target.name}: {dmg} damage (hp now {target.hp}).")
    battle._apply_burn(target, potency=10, count=0, log=log)


def effect_boss_amplitude_cascade(battle, source, target, log):
    dmg = battle._apply_damage(target, 40, log)
    log.add(f"BOSS uses Amplitude Cascade on {target.name}: {dmg} damage (hp now {target.hp}).")
    battle._apply_tremor(target, potency=1, count=1, log=log)
    battle._trigger_amplitude_conversion(target, log)
    battle._tremor_burst(target, log)
