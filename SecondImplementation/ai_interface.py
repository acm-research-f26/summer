"""
Headless, command-driven interface for playing the game programmatically -
no pygame window, no display required at all. This is the counterpart to
manual play (`python game_ui.py`, which opens the visual pygame window):
that's for a human; this module is for a bot/AI/script.

Typical usage:

    from ai_interface import run_headless_battle

    def my_policy(state):
        # state has everything the AI needs: turn number, every unit's hp/
        # status effects/available skills, and the boss's hp/status/chosen
        # skills this turn. See build_state() below for the exact shape.
        commands = []
        for unit in state["units"]:
            if not unit["available_skills"]:
                continue  # dead or staggered - needs no command at all
            commands.append([0, 0])  # bottom skill, attack unopposed
        return commands

    result = run_headless_battle(my_policy, seed=42)
    print(result["outcome"], result["turns"])

The command format for one turn is a list with one [skill_idx, target_idx]
pair per unit that needs an action (i.e. alive AND not staggered - skip any
unit whose "available_skills" list is empty), in the same order as
state["units"]:

    skill_idx:  0 = the BOTTOM available skill, 1 = the TOP available skill.
    target_idx: 0 = attack unopposed.
                1, 2, or 3 = clash the boss's 1st/2nd/3rd chosen skill this
                turn (i.e. boss slot index target_idx - 1). Only valid indices
                that actually exist in state["boss"]["slots"] are usable -
                if the boss has no skills this turn (staggered or dead),
                state["boss"]["slots"] is empty and only target_idx 0 is valid.

Example: [[0, 0], [1, 2], [0, 1]] means the first actionable unit attacks
unopposed with its bottom skill, the second clashes the boss's 2nd chosen
skill with its top skill, and the third clashes the boss's 1st chosen skill
with its bottom skill.

NOTE ON PASSIVES: each unit (and the boss) has a `passive_description` string
in the state - some of these grant a +1.5 roll bonus (and, for one unit, a
damage reduction) under certain conditions, e.g. "if opponent has +15 burn
potency" or "if self has +10 tremor potency". These conditions are checked
against fields already present in state (a unit's own tremor_potency /
burn_potency / magic_bullets, and the boss's tremor_potency / burn_potency),
but the ENGINE - not this interface - is what actually applies the bonus
during clash resolution; nothing needs to be added to your commands for it.
A policy that wants to accurately estimate its own clash odds should account
for these bonuses itself when computing win probabilities, since they matter
a lot in practice (a policy that ignores them scored ~50% in testing; the
same logic aware of passives scored ~75-80%).

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


def _skill_info(unit, skill_type):
    d = unit.get_skill_def(skill_type)
    return {
        "skill_type": skill_type,  # 'skill1' or 'skill2', informational only
        "name": d.name,
        "roll_lo": d.roll_lo,
        "roll_hi": d.roll_hi,
        "base_damage": d.base_damage,
        "description": d.description,
    }


def _combatant_status(entity):
    """Shared status-effect fields for both a PlayerUnit and the Boss."""
    return {
        "hp": entity.hp,
        "max_hp": entity.max_hp,
        "is_alive": entity.is_alive(),
        "is_staggered": entity.is_staggered,
        "stagger_turns_left": entity.stagger_turns_left,
        "active_stagger_thresholds": entity.active_stagger_thresholds(),
        "tremor_potency": entity.tremor_potency,
        "tremor_count": entity.tremor_count,
        "tremor_type": entity.tremor_type,  # "normal" or "scorch"
        "burn_potency": entity.burn_potency,
        "burn_count": entity.burn_count,
        "dark_flame_count": entity.dark_flame_count,
    }


def _unit_info(unit):
    info = {"name": unit.name, "passive_description": unit.passive_description, **_combatant_status(unit)}
    info["magic_bullets"] = unit.magic_bullets
    info["max_magic_bullets"] = unit.max_magic_bullets
    needs_action = unit.is_alive() and not unit.is_staggered
    info["available_skills"] = (
        [_skill_info(unit, st) for st in unit.queue.available] if needs_action else []
    )
    info["next_up_skills"] = (
        [_skill_info(unit, st) for st in unit.queue.next_up] if unit.is_alive() else []
    )
    return info


def _boss_slot_info(slot):
    sd = slot.skill_def
    return {
        "name": sd.name,
        "roll_lo": sd.roll_lo,
        "roll_hi": sd.roll_hi,
        "base_damage": sd.base_damage,
        "hits_all": sd.hits_all,
        "targets": list(slot.target_names),
        "clash_roll_bonus": sd.clash_roll_bonus,
        "clash_damage_multiplier": sd.clash_damage_multiplier,
        "description": sd.description,
    }


def build_state(battle, boss_slots):
    """
    Builds the full state dict a policy function needs to decide a turn's
    commands. Call this AFTER battle.boss_choose_turn() so boss_slots
    reflects what the boss actually picked (or didn't - it can be empty if
    the boss is staggered or dead, or if the whole battle is already over).
    """
    return {
        "turn_number": battle.turn_number + 1,  # the turn about to be played
        "units": [_unit_info(u) for u in battle.units.values()],
        "boss": {
            "name": battle.boss.name,
            "passive_description": battle.boss.passive_description,
            **_combatant_status(battle.boss),
            "slots": [_boss_slot_info(s) for s in boss_slots],
        },
    }


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
