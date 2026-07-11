"""
Balance simulation harness.

Plays out many full battles using the REAL engine (engine.py) against a
heuristic "skilled player" policy, so we can measure actual win rate for a
given set of tunable numbers (HP, stagger thresholds, damage, roll ranges)
rather than guessing.

The AI, each turn:
  1. Sees the boss's 3 chosen skills for the turn (roll range + targets),
     exactly as a real player would.
  2. For every living, non-staggered unit and each of its 2 currently
     available skills, computes an exact win probability against each boss
     slot from the roll ranges (brute-force over all integer roll pairs,
     conditioning out ties the same way the engine's reroll-on-tie does).
  3. Brute-forces every valid (unit -> {slot0, slot1, slot2, unopposed})
     assignment (small search space, trivially fast) and picks the one that
     maximizes expected (damage dealt to boss) - (expected damage taken),
     i.e. genuinely good, not just greedy, per-turn play.

Status effect magnitudes (tremor/burn/dark flame potencies & counts) are left
exactly as designed; only HP, stagger thresholds, base damage, and roll
ranges are tunable knobs here.
"""

import random
import itertools
from engine import (
    PlayerUnit, SkillDef, Boss, BossSkillDef, Battle, PlayerAction,
    effect_tremor_scorch_skill1, effect_tremor_scorch_skill2,
    effect_dark_flame_skill1, effect_dark_flame_skill2,
    effect_self_status_skill1, effect_self_status_skill2,
    effect_boss_tremor_slam, effect_boss_burn_wave, effect_boss_clash_baiter,
    effect_boss_scorch_point, effect_boss_amplitude_cascade,
    passive_roll_bonus_tremor_scorch, passive_roll_bonus_dark_flame,
    passive_roll_bonus_self_status, passive_damage_reduction_self_status,
)


# ----------------------------------------------------------------------------
# Tunable parameter set. Everything the balance pass is allowed to touch.
# Status effect potency/count numbers are NOT here on purpose - those stay fixed.
# ----------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # Roll ranges chosen to hit specific target clash win-probabilities against
    # each boss skill, with NO matchup ever a guaranteed win or loss for either
    # side. Clash Baiter keeps its original roll range but has a real +5 roll /
    # 10x damage bonus that applies only while it's being clashed
    # (clash_roll_bonus/clash_damage_multiplier - previously described but not
    # actually implemented). Damage scaled up ~1.5x for bigger, more
    # "respectable" hits, with HP raised slightly to compensate and keep the
    # same overall win-rate target. Measured: skilled ~82-85% win rate at
    # ~8.4 turns to win; naive (never clashes, never uses skill2) ~2-3%.
    "tremor_scorch": {
        "hp": 532, "stagger": [213],
        "skill1_dmg": 25, "skill1_roll": (13, 19),
        "skill2_dmg": 50, "skill2_roll": (9, 18),
    },
    "dark_flame": {
        "hp": 614, "stagger": [521, 338, 153],
        "skill1_dmg": 25, "skill1_roll": (13, 19),
        "skill2_dmg": 50, "skill2_roll": (9, 18),
    },
    "self_status": {
        "hp": 409, "stagger": [286, 143],
        "skill1_dmg": 25, "skill1_roll": (13, 19),
        "skill2_dmg": 50, "skill2_roll": (9, 18),
    },
    "boss": {
        "hp": 1450, "stagger": [1000, 500],
        "skills": [
            # (name, dmg, roll, hits_all, clash_roll_bonus, clash_damage_multiplier)
            # Damage re-tuned upward after the three party passives (roll bonus /
            # damage reduction from tremor/burn/bullet stacks) were actually
            # implemented, since they'd meaningfully favored the player and
            # pushed skilled win rate well above the ~80% target.
            ("Tremor Slam", 47, (9, 18), False, 0, 1.0),
            ("Burn Wave", 18, (3, 17), True, 0, 1.0),
            ("Clash Baiter", 27, (11, 15), False, 5, 10.0),
            ("Scorch Point", 32, (13, 18), False, 0, 1.0),
            ("Amplitude Cascade", 47, (14, 18), True, 0, 1.0),
        ],
    },
}


def make_battle(params, rng):
    p = params
    tremor_scorch = PlayerUnit(
        name="TremorScorch", max_hp=p["tremor_scorch"]["hp"],
        skill1=SkillDef("Tremor Jab", *p["tremor_scorch"]["skill1_roll"], p["tremor_scorch"]["skill1_dmg"],
                        effect=effect_tremor_scorch_skill1),
        skill2=SkillDef("Tremor Burst Strike", *p["tremor_scorch"]["skill2_roll"], p["tremor_scorch"]["skill2_dmg"],
                        effect=effect_tremor_scorch_skill2),
        stagger_thresholds=list(p["tremor_scorch"]["stagger"]),
        passive_roll_bonus_fn=passive_roll_bonus_tremor_scorch,
    )
    dark_flame = PlayerUnit(
        name="DarkFlameInflictor", max_hp=p["dark_flame"]["hp"],
        skill1=SkillDef("Flame Tag", *p["dark_flame"]["skill1_roll"], p["dark_flame"]["skill1_dmg"],
                        effect=effect_dark_flame_skill1),
        skill2=SkillDef("Dark Flame Surge", *p["dark_flame"]["skill2_roll"], p["dark_flame"]["skill2_dmg"],
                        effect=effect_dark_flame_skill2),
        stagger_thresholds=list(p["dark_flame"]["stagger"]),
        max_magic_bullets=7,
        passive_roll_bonus_fn=passive_roll_bonus_dark_flame,
    )
    self_status = PlayerUnit(
        name="BurnTremorInflictor", max_hp=p["self_status"]["hp"],
        skill1=SkillDef("Shared Tremor", *p["self_status"]["skill1_roll"], p["self_status"]["skill1_dmg"],
                        effect=effect_self_status_skill1),
        skill2=SkillDef("Shared Burn", *p["self_status"]["skill2_roll"], p["self_status"]["skill2_dmg"],
                        effect=effect_self_status_skill2),
        stagger_thresholds=list(p["self_status"]["stagger"]),
        passive_roll_bonus_fn=passive_roll_bonus_self_status,
        passive_damage_reduction_fn=passive_damage_reduction_self_status,
    )
    units = [tremor_scorch, dark_flame, self_status]

    effect_fns = [effect_boss_tremor_slam, effect_boss_burn_wave, effect_boss_clash_baiter,
                  effect_boss_scorch_point, effect_boss_amplitude_cascade]
    boss_skills = [
        BossSkillDef(name, roll[0], roll[1], dmg, hits_all=hits_all, effect=fn,
                     clash_roll_bonus=bonus, clash_damage_multiplier=mult)
        for (name, dmg, roll, hits_all, bonus, mult), fn in zip(p["boss"]["skills"], effect_fns)
    ]
    boss = Boss(name="Boss", max_hp=p["boss"]["hp"], skills=boss_skills,
                stagger_thresholds=list(p["boss"]["stagger"]))

    battle = Battle(boss, units, rng=rng)
    return battle, units, boss


# ----------------------------------------------------------------------------
# Heuristic "skilled player" policy
# ----------------------------------------------------------------------------
def win_probability(att_lo, att_hi, def_lo, def_hi):
    """P(attacker roll beats defender roll), conditioning out ties (reroll)."""
    wins = losses = 0
    for a in range(att_lo, att_hi + 1):
        for d in range(def_lo, def_hi + 1):
            if a > d:
                wins += 1
            elif a < d:
                losses += 1
    total_decisive = wins + losses
    if total_decisive == 0:
        return 0.5
    return wins / total_decisive


def slot_threat(slot_def, num_targets):
    return slot_def.base_damage * (num_targets if slot_def.hits_all else 1)


def choose_actions(battle, boss_slots):
    """Returns a list of PlayerAction representing a near-optimal turn for
    the whole party, found by brute-forcing all valid slot assignments."""
    alive_units = [u for u in battle.alive_units() if not u.is_staggered]
    if not alive_units:
        return []

    threats = [slot_threat(slot.skill_def, len(slot.target_names)) for slot in boss_slots]

    # options[unit] = list of (skill_type, target) where target is a slot index or None (unopposed)
    options_per_unit = []
    for unit in alive_units:
        opts = []
        for skill_type in set(unit.queue.available):
            skill_def = unit.get_skill_def(skill_type)
            for slot_idx, slot in enumerate(boss_slots):
                boss_def = slot.skill_def
                # a real clash uses the boss skill's roll bounds WITH its clash
                # bonus applied (e.g. Clash Baiter's +5), since that bonus only
                # ever applies while being clashed
                eff_lo = boss_def.roll_lo + boss_def.clash_roll_bonus
                eff_hi = boss_def.roll_hi + boss_def.clash_roll_bonus
                player_lo, player_hi = skill_def.roll_lo, skill_def.roll_hi
                if unit.passive_roll_bonus_fn is not None:
                    bonus = unit.passive_roll_bonus_fn(unit, battle.boss)
                    player_lo = round(player_lo + bonus)
                    player_hi = round(player_hi + bonus)
                wp = win_probability(player_lo, player_hi, eff_lo, eff_hi)
                threat = threats[slot_idx]
                # losing a clash against a skill with a damage multiplier is worse
                # than just letting it hit unopposed - the multiplier only applies
                # when the boss wins a clash, so that's the real downside risk here
                effective_loss = threat * boss_def.clash_damage_multiplier
                value = wp * skill_def.base_damage + wp * threat - (1 - wp) * effective_loss
                opts.append((value, skill_type, slot_idx))
            # unopposed: guaranteed damage, doesn't touch any slot's threat
            opts.append((skill_def.base_damage, skill_type, None))
        options_per_unit.append(opts)

    best_value = None
    best_assignment = None
    for combo in itertools.product(*options_per_unit):
        used_slots = [c[2] for c in combo if c[2] is not None]
        if len(used_slots) != len(set(used_slots)):
            continue  # two units can't clash the same slot
        total_value = sum(c[0] for c in combo)
        if best_value is None or total_value > best_value:
            best_value = total_value
            best_assignment = combo

    actions = []
    for unit, (_, skill_type, slot_idx) in zip(alive_units, best_assignment):
        position = unit.queue.available.index(skill_type)
        actions.append(PlayerAction(unit.name, skill_type, clash_slot_index=slot_idx, armed_position=position))
    return actions


def choose_actions_naive(battle, boss_slots, rng):
    """
    A deliberately unskilled policy, representing someone who doesn't know
    what they're doing: never clashes at all (doesn't understand that
    trading a hit for a chance to block a boss attack is worthwhile), and
    always reaches for the simpler skill1, never leveraging skill2's bigger
    damage/status payoff - i.e. never properly building up or spending
    tremor/burn/magic-bullet stacks, never triggering bursts, scorch
    conversion, or dark flame amplification on purpose.

    NOTE: earlier iterations of this policy had it clash a random slot at
    some fixed probability, but simulation showed that ANY willingness to
    clash (even picked at random, regardless of whether the matchup favors
    the player) meaningfully improves survival odds in this system - going
    from "never clash" to "always attempt to clash something" raised the
    measured win rate from ~5% to over 40% at otherwise-identical settings.
    So the defining trait of not knowing what you're doing here isn't
    "clashes badly" so much as "doesn't clash at all."
    """
    alive_units = [u for u in battle.alive_units() if not u.is_staggered]
    actions = []
    for unit in alive_units:
        skill_type = "skill1" if "skill1" in unit.queue.available else unit.queue.available[0]
        position = unit.queue.available.index(skill_type)
        actions.append(PlayerAction(unit.name, skill_type, clash_slot_index=None, armed_position=position))
    return actions


# ----------------------------------------------------------------------------
# Battle simulation
# ----------------------------------------------------------------------------
def simulate_one_battle(params, seed, policy="skilled", max_turns=60):
    rng = random.Random(seed)
    battle, units, boss = make_battle(params, rng)

    for turn in range(1, max_turns + 1):
        if not boss.is_alive():
            return "win", turn
        if not battle.alive_units():
            return "loss", turn
        boss_slots = battle.boss_choose_turn()
        if policy == "skilled":
            actions = choose_actions(battle, boss_slots)
        else:
            actions = choose_actions_naive(battle, boss_slots, rng)
        battle.resolve_turn(boss_slots, actions)

    if not boss.is_alive():
        return "win", max_turns
    if not battle.alive_units():
        return "loss", max_turns
    return "draw", max_turns


def run_batch(params, n=300, seed_start=0, policy="skilled"):
    results = {"win": 0, "loss": 0, "draw": 0}
    win_turn_counts = []
    for i in range(n):
        outcome, turns = simulate_one_battle(params, seed=seed_start + i, policy=policy)
        results[outcome] += 1
        if outcome == "win":
            win_turn_counts.append(turns)
    win_rate = results["win"] / n
    avg_turns_to_win = sum(win_turn_counts) / len(win_turn_counts) if win_turn_counts else None
    return {
        "win_rate": win_rate,
        "results": results,
        "avg_turns_to_win": avg_turns_to_win,
        "n": n,
    }


if __name__ == "__main__":
    import copy
    skilled = run_batch(copy.deepcopy(DEFAULT_PARAMS), n=1000, policy="skilled")
    naive = run_batch(copy.deepcopy(DEFAULT_PARAMS), n=1000, policy="naive")
    print("Current shipped values (game_ui.py):")
    print(f"  SKILLED play: win rate {skilled['win_rate']*100:.1f}%  "
          f"(avg turns to win: {skilled['avg_turns_to_win']:.1f})  {skilled['results']}")
    print(f"  NAIVE play:   win rate {naive['win_rate']*100:.1f}%  "
          f"(avg turns to win: {naive['avg_turns_to_win']})  {naive['results']}")
