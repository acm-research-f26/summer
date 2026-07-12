"""
Headless, command-driven interface for playing the game programmatically -
no pygame window, no display required at all. This is the counterpart to
manual play (`python game_ui.py`, which opens the visual pygame window):
that's for a human; this module is for a bot/AI/script.

Typical usage:

    from ai_interface import run_headless_battle

    def my_policy(state):
        commands = []
        for unit in state["units"]:
            if unit["is_dead"] or unit["is_staggered"]:
                continue  # needs no command at all
            commands.append([0, 0])  # bottom skill, attack unopposed
        return commands

    result = run_headless_battle(my_policy, seed=42)
    print(result["outcome"], result["turns"])

STATE SHAPE. `state` is deliberately small and mostly-numeric (built for
feeding a model, not for reading prose) - see build_state()'s docstring for
the exact shape, but in short:

    state["turn_number"]  - int
    state["units"]        - always exactly 3 entries (one per party unit, in
                             a FIXED order that never changes across the
                             whole battle - index 0 always means the same
                             unit, etc), each with hp / is_dead / is_staggered
                             / burn & tremor potency+count / whether tremor
                             scorch is genuinely active / magic bullets /
                             clash_power_level / damage_reduction_active /
                             a 4-int "skills" array (current bottom+top, next
                             bottom+top, each 0=skill1 or 1=skill2), and how
                             many stagger thresholds it has left (just the
                             count - not where they are).
    state["boss"]         - same status fields as a unit, PLUS
                             "chosen_skills": always exactly 3 [skill_number,
                             target] pairs (see below), [-1, -1] for any slot
                             the boss didn't actually use this turn (it was
                             staggered or dead).

Roll ranges, base damage, and which skills hit all 3 party members at once
are FIXED - they never change turn to turn - so they're not repeated in
`state` every turn. Fetch them once with get_skill_catalog() instead.

skill_number (0-4) identifies which of the boss's 5 possible skills a
chosen_skills entry is, matching the order in get_skill_catalog()["boss_skills"].
target is 0/1/2 for a specific party unit (same fixed indexing as state["units"]),
or -1 if that skill hits all 3 at once (an AoE skill - get_skill_catalog()
tells you which skill_numbers are AoE via "hits_all").

The command format for one turn is a list with one [skill_idx, target_idx]
pair per unit that needs an action (i.e. not dead and not staggered - skip
any other unit), in the same fixed order as state["units"]:

    skill_idx:  0 = the BOTTOM available skill, 1 = the TOP available skill.
    target_idx: 0 = attack unopposed.
                1, 2, or 3 = clash the boss's 1st/2nd/3rd chosen skill this
                turn (i.e. chosen_skills index target_idx - 1). Only valid
                indices that actually exist this turn are usable - if the
                boss has no real skills this turn (staggered or dead, so its
                chosen_skills are all [-1, -1]), only target_idx 0 is valid.

Example: [[0, 0], [1, 2], [0, 1]] means the first actionable unit attacks
unopposed with its bottom skill, the second clashes the boss's 2nd chosen
skill with its top skill, and the third clashes the boss's 1st chosen skill
with its bottom skill.

NOTE ON PASSIVES: `clash_power_level` (0, 1, or 2) tells you how many of a
unit's passive +1.5-roll conditions are CURRENTLY satisfied (each one always
grants exactly +1.5, so 2 active conditions = +3.0 total) - the engine
applies this automatically during clash resolution, so nothing needs to be
added to your commands for it, but a policy that wants to accurately
estimate its own clash odds should account for it when computing win
probabilities, since it matters a lot in practice (a policy that ignores it
scored ~50% in testing; the same logic aware of it scored ~75-80%).
`damage_reduction_active` similarly tells you whether a unit's passive
damage reduction (only one unit has this) is currently in effect.

LOOKING AHEAD (simulating a few turns before committing to a real decision):
`state` is just a read-only snapshot, so it alone can't tell you "what would
happen if I tried this?" - for that you need the actual, live `Battle`
object, which has a `.clone()` method for exactly this purpose (see its
docstring in engine.py). If your policy function accepts extra parameters,
`run_headless_battle` passes it the real `Battle` and the boss's already-
chosen `boss_slots` for the turn about to be played, so you can clone the
battle, try out a candidate move AGAINST THAT SAME boss turn, look at the
result, and repeat before returning your actual commands:

    def lookahead_policy(state, battle, boss_slots):
        # IMPORTANT: reuse the given boss_slots on your clones - don't call
        # scratch.boss_choose_turn() yourself, since that would re-roll a
        # DIFFERENT random boss turn than the one that's actually happening.
        best_commands, best_score = None, None
        for candidate in generate_some_candidate_command_lists(state):
            scratch = battle.clone()  # throwaway copy, safe to mess with
            try:
                actions = commands_to_player_actions(scratch, boss_slots, candidate)
            except ValueError:
                continue  # e.g. two units clashing the same slot - just skip it
            scratch.resolve_turn(boss_slots, actions)
            score = scratch.boss.hp  # or whatever you want to optimize
            if best_score is None or score < best_score:
                best_score, best_commands = score, candidate
        return best_commands

    run_headless_battle(lookahead_policy, seed=1)

A policy that only accepts `state` (the simple case at the top of this
docstring) still works exactly as before - `run_headless_battle` detects how
many parameters your function takes and calls it with just as many
arguments: `(state)`, `(state, battle)`, or `(state, battle, boss_slots)`.
"""

import inspect
import random

from engine import Battle, PlayerAction
from game_data import build_party_units, build_boss


def _policy_arg_count(policy_fn):
    """How many positional arguments policy_fn's signature accepts (1-3)."""
    try:
        return len(inspect.signature(policy_fn).parameters)
    except (TypeError, ValueError):
        # some callables (e.g. certain builtins) don't support introspection -
        # just assume the simple, one-argument form in that case.
        return 1


def _status_fields(entity, opponent):
    """
    Shared compact status fields for both a PlayerUnit and the Boss.

    `opponent` is whoever's "opponent has +X potency"-style passive
    conditions should be checked against (the boss, for a party unit; None
    for the boss itself, since the boss has no passives of its own).

    tremor_scorch_active is deliberately false whenever tremor_count is 0,
    even if `tremor_type` still technically says "scorch" from a
    not-yet-expired amplitude conversion - with no count left there's no
    tremor stack at all (scorch or otherwise) for it to matter.
    """
    clash_power_level = 0
    if getattr(entity, "passive_roll_bonus_fn", None) is not None and opponent is not None:
        bonus = entity.passive_roll_bonus_fn(entity, opponent)
        # each satisfied condition always grants exactly +1.5, so dividing
        # tells us how many of the (up to 2) conditions are active: 0, 1, or 2.
        clash_power_level = round(bonus / 1.5) if bonus else 0

    damage_reduction_active = False
    if getattr(entity, "passive_damage_reduction_fn", None) is not None:
        if entity.passive_damage_reduction_fn(entity) == 1.0:
            damage_reduction_active = 0
        elif entity.passive_damage_reduction_fn(entity) == 0.8:
            damage_reduction_active = 1
        else:
            damage_reduction_active = 2

    return {
        "hp": entity.hp,
        "is_dead": not entity.is_alive(),
        "is_staggered": entity.is_staggered,
        "num_stagger_thresholds_remaining": len(entity.active_stagger_thresholds()),
        "burn_potency": entity.burn_potency,
        "burn_count": entity.burn_count,
        "tremor_potency": entity.tremor_potency,
        "tremor_count": entity.tremor_count,
        "tremor_scorch_active": entity.tremor_count > 0 and entity.tremor_type == "scorch",
        "magic_bullets": entity.magic_bullets,
        "clash_power_level": clash_power_level,
        "damage_reduction_active": damage_reduction_active,
    }


def _skill_bits(unit):
    """[current_bottom, current_top, next_bottom, next_top], each 0 (skill1) or 1 (skill2)."""
    skill_types = list(unit.queue.available) + list(unit.queue.next_up)
    return [0 if skill_type == "skill1" else 1 for skill_type in skill_types]


def build_state(battle, boss_slots):
    """
    Builds the compact state dict a policy function needs to decide a turn's
    commands - see the module docstring for the full shape and field
    meanings. Call this AFTER battle.boss_choose_turn(), so boss_slots
    reflects what the boss actually picked this turn (it can be an empty
    list if the boss is staggered or dead, or the whole battle is already over).
    """
    # a stable, fixed ordering: index 0/1/2 always refer to the same unit for
    # the entire battle, regardless of who's alive/dead/staggered right now.
    unit_order = list(battle.units.keys())

    units_state = []
    for name in unit_order:
        unit = battle.units[name]
        entry = _status_fields(unit, opponent=battle.boss)
        entry["skills"] = _skill_bits(unit)
        units_state.append(entry)

    chosen_skills = []
    for slot in boss_slots:
        skill_number = battle.boss.skills.index(slot.skill_def)
        target = -1 if slot.skill_def.hits_all else unit_order.index(slot.target_names[0])
        chosen_skills.append([skill_number, target])
    while len(chosen_skills) < 3:
        chosen_skills.append([-1, -1])  # boss didn't actually use this slot this turn

    boss_state = _status_fields(battle.boss, opponent=None)
    boss_state["chosen_skills"] = chosen_skills

    return {
        "turn_number": battle.turn_number + 1,
        "units": units_state,
        "boss": boss_state,
    }


def get_skill_catalog(battle):
    """
    Static reference info that never changes turn to turn - roll ranges,
    base damage, which skills hit all 3 party members at once, and the
    boss's clash-bonus values - keyed the same way state's skill_number /
    unit-index encoding is, so you only need to fetch this once per battle
    (not every turn) and cross-reference it against the per-turn state.
    """
    boss_skills = []
    for skill_number, skill_def in enumerate(battle.boss.skills):
        boss_skills.append({
            "skill_number": skill_number,
            "name": skill_def.name,
            "roll_lo": skill_def.roll_lo,
            "roll_hi": skill_def.roll_hi,
            "base_damage": skill_def.base_damage,
            "hits_all": skill_def.hits_all,
            "clash_roll_bonus": skill_def.clash_roll_bonus,
            "clash_damage_multiplier": skill_def.clash_damage_multiplier,
        })

    party_units = []
    for name in battle.units:
        unit = battle.units[name]
        party_units.append({
            "name": name,
            "skill1": {"name": unit.skill1.name, "roll_lo": unit.skill1.roll_lo,
                       "roll_hi": unit.skill1.roll_hi, "base_damage": unit.skill1.base_damage},
            "skill2": {"name": unit.skill2.name, "roll_lo": unit.skill2.roll_lo,
                       "roll_hi": unit.skill2.roll_hi, "base_damage": unit.skill2.base_damage},
        })

    return {"boss_skills": boss_skills, "party_units": party_units}


def commands_to_player_actions(battle, boss_slots, commands):
    """
    Converts the compact [[skill_idx, target_idx], ...] command format into
    real PlayerAction objects, one per unit that's alive and not staggered
    (in that same deployment order), validating the input so a bad command
    raises a clear error instead of silently dropping a unit's action.
    """
    actionable_units = [u for u in battle.units.values() if u.is_alive() and not u.is_staggered]

    if len(commands) != len(actionable_units):
        raise ValueError(
            f"Expected {len(actionable_units)} command(s), one per actionable unit "
            f"({[u.name for u in actionable_units]}), but got {len(commands)}."
        )

    num_slots = len(boss_slots)
    claimed_slots = {}
    actions = []
    for unit, command in zip(actionable_units, commands):
        if len(command) != 2:
            raise ValueError(f"Each command must be [skill_idx, target_idx], got {command!r}.")
        skill_idx, target_idx = command
        if skill_idx not in (0, 1):
            raise ValueError(f"skill_idx must be 0 (bottom) or 1 (top), got {skill_idx!r}.")
        if not (0 <= target_idx <= 3):
            raise ValueError(f"target_idx must be 0-3, got {target_idx!r}.")

        if target_idx == 0:
            clash_slot_index = None
        else:
            clash_slot_index = target_idx - 1
            if clash_slot_index >= num_slots:
                raise ValueError(
                    f"target_idx {target_idx} refers to boss slot {clash_slot_index}, "
                    f"but the boss only has {num_slots} slot(s) this turn."
                )
            if clash_slot_index in claimed_slots:
                raise ValueError(
                    f"Both {claimed_slots[clash_slot_index]} and {unit.name} were given "
                    f"target_idx {target_idx} - only one unit may clash a given boss slot."
                )
            claimed_slots[clash_slot_index] = unit.name

        skill_type = unit.queue.available[skill_idx]
        actions.append(PlayerAction(unit.name, skill_type, clash_slot_index=clash_slot_index,
                                     armed_position=skill_idx))
    return actions


def run_headless_battle(policy_fn, max_turns=60, seed=None, on_turn_end=None):
    """
    Fully headless battle runner - no pygame, no display. Calls policy_fn
    once per turn and expects the compact command list back (see module
    docstring). Returns {"outcome": "win"|"loss"|"draw", "turns": int, "battle": Battle}.

    policy_fn can accept 1, 2, or 3 positional arguments:
        policy_fn(state)                       - the simple case
        policy_fn(state, battle)               - also gets the live Battle,
                                                  e.g. to battle.clone() for lookahead
        policy_fn(state, battle, boss_slots)   - also gets the boss's already-
                                                  chosen skills for this turn, so
                                                  lookahead code can resolve clones
                                                  against that SAME boss turn
                                                  instead of re-rolling a new one
    (see the "LOOKING AHEAD" section of the module docstring for a full example).

    on_turn_end(state, commands, log), if given, is called after each turn
    resolves (log is the engine's BattleLog for that turn) - handy for
    printing/inspecting what happened without needing to re-derive it.
    """
    rng = random.Random(seed)
    units = build_party_units()
    boss = build_boss()
    battle = Battle(boss, units, rng=rng)
    arg_count = _policy_arg_count(policy_fn)

    for _ in range(max_turns):
        if not battle.boss.is_alive() or not battle.alive_units():
            break
        boss_slots = battle.boss_choose_turn()
        state = build_state(battle, boss_slots)

        if arg_count >= 3:
            commands = policy_fn(state, battle, boss_slots)
        elif arg_count == 2:
            commands = policy_fn(state, battle)
        else:
            commands = policy_fn(state)

        actions = commands_to_player_actions(battle, boss_slots, commands)
        log = battle.resolve_turn(boss_slots, actions)
        if on_turn_end is not None:
            on_turn_end(state, commands, log)

    if not battle.boss.is_alive():
        outcome = "win"
    elif not battle.alive_units():
        outcome = "loss"
    else:
        outcome = "draw"  # hit max_turns without either side being defeated
    return {"outcome": outcome, "turns": battle.turn_number, "battle": battle}
