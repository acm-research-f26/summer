import random
from engine import Battle, PlannedBossSkill
from game_ui import make_units, make_boss
from ai_interface import build_state, commands_to_player_actions, run_headless_battle


def test_build_state_shape():
    rng = random.Random(0)
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=rng)
    boss_slots = battle.boss_choose_turn()

    state = build_state(battle, boss_slots)
    assert state["turn_number"] == 1
    assert len(state["units"]) == 3
    for u in state["units"]:
        assert "name" in u and "hp" in u and "is_staggered" in u
        assert "passive_description" in u
        assert len(u["available_skills"]) == 2  # alive, unstaggered -> bottom+top
        assert len(u["next_up_skills"]) == 2
    assert state["boss"]["name"] == boss.name
    assert len(state["boss"]["slots"]) == 3
    for slot in state["boss"]["slots"]:
        assert "roll_lo" in slot and "roll_hi" in slot and "hits_all" in slot
    print("test_build_state_shape passed.")


def test_available_skills_empty_for_staggered_unit():
    rng = random.Random(0)
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=rng)
    units[0].is_staggered = True
    units[0].stagger_turns_left = 1
    boss_slots = battle.boss_choose_turn()

    state = build_state(battle, boss_slots)
    staggered_info = next(u for u in state["units"] if u["name"] == units[0].name)
    assert staggered_info["available_skills"] == []
    assert staggered_info["is_staggered"] is True
    print("test_available_skills_empty_for_staggered_unit passed.")


def test_boss_slots_empty_when_boss_staggered():
    rng = random.Random(0)
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=rng)
    boss.is_staggered = True
    boss.stagger_turns_left = 1
    boss_slots = battle.boss_choose_turn()
    assert boss_slots == []

    state = build_state(battle, boss_slots)
    assert state["boss"]["slots"] == []
    assert state["boss"]["is_staggered"] is True
    print("test_boss_slots_empty_when_boss_staggered passed.")


def test_commands_to_player_actions_basic_mapping():
    rng = random.Random(0)
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=rng)
    boss_slots = battle.boss_choose_turn()

    # unit 0: bottom skill, unopposed. unit 1: top skill, clash slot 2 (target_idx=3).
    # unit 2: bottom skill, clash slot 0 (target_idx=1).
    commands = [[0, 0], [1, 3], [0, 1]]
    actions = commands_to_player_actions(battle, boss_slots, commands)

    assert len(actions) == 3
    a0, a1, a2 = actions
    assert a0.unit_name == units[0].name and a0.clash_slot_index is None and a0.armed_position == 0
    assert a1.unit_name == units[1].name and a1.clash_slot_index == 2 and a1.armed_position == 1
    assert a2.unit_name == units[2].name and a2.clash_slot_index == 0 and a2.armed_position == 0
    print("test_commands_to_player_actions_basic_mapping passed.")


def test_commands_to_player_actions_skips_staggered_units():
    rng = random.Random(0)
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=rng)
    units[1].is_staggered = True
    units[1].stagger_turns_left = 1
    boss_slots = battle.boss_choose_turn()

    # only 2 commands needed now (unit[1] is staggered and needs none)
    commands = [[0, 0], [0, 0]]
    actions = commands_to_player_actions(battle, boss_slots, commands)
    assert len(actions) == 2
    assert {a.unit_name for a in actions} == {units[0].name, units[2].name}
    print("test_commands_to_player_actions_skips_staggered_units passed.")


def test_commands_to_player_actions_rejects_wrong_length():
    rng = random.Random(0)
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=rng)
    boss_slots = battle.boss_choose_turn()

    try:
        commands_to_player_actions(battle, boss_slots, [[0, 0], [0, 0]])  # only 2, need 3
        assert False, "should have raised"
    except ValueError as e:
        assert "Expected 3 command" in str(e)
    print("test_commands_to_player_actions_rejects_wrong_length passed.")


def test_commands_to_player_actions_rejects_invalid_skill_idx():
    rng = random.Random(0)
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=rng)
    boss_slots = battle.boss_choose_turn()

    try:
        commands_to_player_actions(battle, boss_slots, [[2, 0], [0, 0], [0, 0]])
        assert False, "should have raised"
    except ValueError as e:
        assert "skill_idx" in str(e)
    print("test_commands_to_player_actions_rejects_invalid_skill_idx passed.")


def test_commands_to_player_actions_rejects_out_of_range_target():
    rng = random.Random(0)
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=rng)
    boss.is_staggered = True  # forces 0 boss slots this turn
    boss_slots = battle.boss_choose_turn()
    assert boss_slots == []

    try:
        commands_to_player_actions(battle, boss_slots, [[0, 1], [0, 0], [0, 0]])  # slot 1 doesn't exist
        assert False, "should have raised"
    except ValueError as e:
        assert "only has 0 slot" in str(e)
    print("test_commands_to_player_actions_rejects_out_of_range_target passed.")


def test_commands_to_player_actions_rejects_duplicate_slot_claim():
    rng = random.Random(0)
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=rng)
    boss_slots = battle.boss_choose_turn()

    try:
        commands_to_player_actions(battle, boss_slots, [[0, 1], [0, 1], [0, 0]])  # both claim slot 0
        assert False, "should have raised"
    except ValueError as e:
        assert "only one unit may clash" in str(e)
    print("test_commands_to_player_actions_rejects_duplicate_slot_claim passed.")


def test_run_headless_battle_end_to_end_with_trivial_policy():
    def trivial_policy(state):
        return [[0, 0] for u in state["units"] if u["available_skills"]]

    result = run_headless_battle(trivial_policy, max_turns=60, seed=1)
    assert result["outcome"] in ("win", "loss", "draw")
    assert result["turns"] > 0
    print(f"test_run_headless_battle_end_to_end_with_trivial_policy passed. "
          f"(outcome={result['outcome']}, turns={result['turns']})")


def test_run_headless_battle_calls_on_turn_end_callback():
    calls = []

    def trivial_policy(state):
        return [[0, 0] for u in state["units"] if u["available_skills"]]

    def on_turn_end(state, commands, log):
        calls.append((state["turn_number"], commands))

    run_headless_battle(trivial_policy, max_turns=5, seed=1, on_turn_end=on_turn_end)
    assert len(calls) > 0
    assert calls[0][0] == 1
    print("test_run_headless_battle_calls_on_turn_end_callback passed.")


def test_run_headless_battle_is_deterministic_given_same_seed():
    def trivial_policy(state):
        return [[0, 0] for u in state["units"] if u["available_skills"]]

    r1 = run_headless_battle(trivial_policy, seed=7)
    r2 = run_headless_battle(trivial_policy, seed=7)
    assert r1["outcome"] == r2["outcome"]
    assert r1["turns"] == r2["turns"]
    print("test_run_headless_battle_is_deterministic_given_same_seed passed.")


def test_target_idx_matches_boss_slot_target_names():
    """Sanity check that target_idx N really does clash the boss slot the
    state described as slots[N-1], by checking who gets retargeted."""
    rng = random.Random(3)
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=rng)

    # hand-construct boss slots so we know exactly what's in each
    filler = boss.skills[0]
    boss_slots = [
        PlannedBossSkill(filler, [units[1].name]),
        PlannedBossSkill(filler, [units[1].name]),
        PlannedBossSkill(filler, [units[1].name]),
    ]
    state = build_state(battle, boss_slots)
    assert len(state["boss"]["slots"]) == 3

    # unit[0] clashes target_idx=2, which should map to boss_slots[1]
    commands = [[0, 2], [0, 0], [0, 0]]
    actions = commands_to_player_actions(battle, boss_slots, commands)
    a0 = next(a for a in actions if a.unit_name == units[0].name)
    assert a0.clash_slot_index == 1
    print("test_target_idx_matches_boss_slot_target_names passed.")


if __name__ == "__main__":
    test_build_state_shape()
    test_available_skills_empty_for_staggered_unit()
    test_boss_slots_empty_when_boss_staggered()
    test_commands_to_player_actions_basic_mapping()
    test_commands_to_player_actions_skips_staggered_units()
    test_commands_to_player_actions_rejects_wrong_length()
    test_commands_to_player_actions_rejects_invalid_skill_idx()
    test_commands_to_player_actions_rejects_out_of_range_target()
    test_commands_to_player_actions_rejects_duplicate_slot_claim()
    test_run_headless_battle_end_to_end_with_trivial_policy()
    test_run_headless_battle_calls_on_turn_end_callback()
    test_run_headless_battle_is_deterministic_given_same_seed()
    test_target_idx_matches_boss_slot_target_names()
    print("\nALL AI INTERFACE TESTS PASSED")
