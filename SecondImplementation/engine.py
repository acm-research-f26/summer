import random
from dataclasses import dataclass, field
from skill_queue import SkillQueue


@dataclass
class SkillDef:
    name: str
    roll_lo: int
    roll_hi: int
    base_damage: int


@dataclass
class PlayerUnit:
    name: str
    max_hp: int
    skill1: SkillDef
    skill2: SkillDef
    hp: int = field(init=False)
    queue: SkillQueue = field(init=False)

    def __post_init__(self):
        self.hp = self.max_hp

    def init_queue(self, rng):
        self.queue = SkillQueue(rng)

    def get_skill_def(self, skill_type):
        return self.skill1 if skill_type == 'skill1' else self.skill2

    def is_alive(self):
        return self.hp > 0


@dataclass
class BossSkillDef:
    name: str
    roll_lo: int
    roll_hi: int
    base_damage: int
    hits_all: bool = False


@dataclass
class Boss:
    name: str
    max_hp: int
    skills: list  # list of 5 BossSkillDef
    hp: int = field(init=False)

    def __post_init__(self):
        self.hp = self.max_hp

    def is_alive(self):
        return self.hp > 0


class PlannedBossSkill:
    """One of the boss's 3 chosen skills for this turn, plus its target(s)."""
    def __init__(self, skill_def, target_names):
        self.skill_def = skill_def
        self.target_names = target_names  # list of unit names (usually 1, or all for hits_all)
        self.clashed_by = None  # will hold (unit_name, skill_type) if a player clashes it
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
        Returns a BattleLog.
        """
        log = BattleLog()
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
            if clashing_action is not None:
                self._resolve_clash(slot, clashing_action, log)
            else:
                self._execute_boss_skill(slot, log)

        # Step 2: remaining (unopposed) player actions, in deployment order
        # (deployment order = the order units were passed into Battle(), i.e. dict insertion order)
        for action in player_actions:
            if action.discarded:
                continue
            if action.clash_slot_index is not None:
                # already resolved during clash step (whether it won or lost)
                continue
            self._execute_player_skill(action, log, target_name=self.boss.name)

        return log

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
            self._execute_boss_skill(boss_slot, log)
        else:
            log.add(f"  {unit.name} wins clash. {boss_def.name} is discarded.")
            boss_slot.discarded = True
            self._execute_player_skill(player_action, log, target_name=self.boss.name)

    def _execute_boss_skill(self, slot, log):
        if slot.discarded:
            return
        skill_def = slot.skill_def
        for target_name in slot.target_names:
            target = self.get_unit(target_name)
            if not target.is_alive():
                continue
            target.hp -= skill_def.base_damage
            log.add(
                f"BOSS uses {skill_def.name} on {target_name}: "
                f"{skill_def.base_damage} damage (hp now {target.hp})"
            )

    def _execute_player_skill(self, action, log, target_name):
        if action.discarded:
            return
        unit = self.get_unit(action.unit_name)
        skill_def = unit.get_skill_def(action.skill_type)
        self.boss.hp -= skill_def.base_damage
        log.add(
            f"{unit.name} uses {skill_def.name}: "
            f"{skill_def.base_damage} damage to {self.boss.name} (boss hp now {self.boss.hp})"
        )
        # advance this unit's skill queue now that the skill has been used
        unit.queue.pick(action.skill_type, position=action.armed_position)
