import random
from engine import (
    PlayerUnit, SkillDef, Boss, BossSkillDef, Battle, PlayerAction, PlannedBossSkill,
)


def make_units():
    return [
        PlayerUnit(name="A", max_hp=100, skill1=SkillDef("S1", 14, 20, 25), skill2=SkillDef("S2", 10, 16, 50)),
        PlayerUnit(name="B", max_hp=100, skill1=SkillDef("S1", 14, 20, 25), skill2=SkillDef("S2", 10, 16, 50)),
        PlayerUnit(name="C", max_hp=100, skill1=SkillDef("S1", 14, 20, 25), skill2=SkillDef("S2", 10, 16, 50)),
    ]


def make_boss():
    skills = [
        BossSkillDef("B1", 12, 16, 25),
        BossSkillDef("B2", 10, 13, 10, hits_all=True),
        BossSkillDef("B3", 10, 14, 15),
        BossSkillDef("B4", 15, 18, 20),
        BossSkillDef("B5", 18, 20, 40, hits_all=True),
    ]
    return Boss(name="Boss", max_hp=1500, skills=skills)


def test_steps_cover_all_boss_slots_and_unopposed_actions():
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=random.Random(1))

    slots = [
        PlannedBossSkill(boss.skills[0], ["A"]),
        PlannedBossSkill(boss.skills[2], ["B"]),
        PlannedBossSkill(boss.skills[3], ["C"]),
    ]
    actions = [
        PlayerAction("A", units[0].queue.available[0], clash_slot_index=0, armed_position=0),
        PlayerAction("B", units[1].queue.available[0], clash_slot_index=None, armed_position=0),
        PlayerAction("C", units[2].queue.available[0], clash_slot_index=None, armed_position=0),
    ]

    steps = list(battle.resolve_turn_steps(slots, actions))
    kinds = [s.kind for s in steps]
    # 3 boss slots (1 clash + 2 boss_unopposed) + 2 player_unopposed (B, C) + 1 turn_end
    assert kinds.count("clash") == 1
    assert kinds.count("boss_unopposed") == 2
    assert kinds.count("player_unopposed") == 2
    assert kinds[-1] == "turn_end"
    assert len(steps) == 6
    print("test_steps_cover_all_boss_slots_and_unopposed_actions passed.")


def test_clash_step_has_roll_and_winner_info():
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=random.Random(2))

    slots = [PlannedBossSkill(boss.skills[0], ["A"])]
    actions = [PlayerAction("A", units[0].queue.available[0], clash_slot_index=0, armed_position=0)]

    steps = list(battle.resolve_turn_steps(slots, actions))
    clash_step = next(s for s in steps if s.kind == "clash")
    assert clash_step.boss_roll is not None
    assert clash_step.player_roll is not None
    assert clash_step.winner in ("boss", "player")
    assert clash_step.boss_skill_name == "B1"
    assert clash_step.participants == ["Boss", "A"]
    assert clash_step.slot_index == 0
    print("test_clash_step_has_roll_and_winner_info passed.")


def test_boss_unopposed_step_participants_include_all_aoe_targets():
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=random.Random(3))

    slots = [PlannedBossSkill(boss.skills[1], ["A", "B", "C"])]  # B2 is hits_all
    actions = [
        PlayerAction("A", units[0].queue.available[0], clash_slot_index=None, armed_position=0),
        PlayerAction("B", units[1].queue.available[0], clash_slot_index=None, armed_position=0),
        PlayerAction("C", units[2].queue.available[0], clash_slot_index=None, armed_position=0),
    ]

    steps = list(battle.resolve_turn_steps(slots, actions))
    boss_step = next(s for s in steps if s.kind == "boss_unopposed")
    assert set(boss_step.participants) == {"Boss", "A", "B", "C"}
    print("test_boss_unopposed_step_participants_include_all_aoe_targets passed.")


def test_resolve_turn_and_resolve_turn_steps_produce_identical_final_state():
    """Same seed, same inputs -> resolve_turn (all-at-once) and resolve_turn_steps
    (drained fully) should leave the battle in identical final state."""
    def run(via_steps):
        units = make_units()
        boss = make_boss()
        battle = Battle(boss, units, rng=random.Random(42))
        slots = [
            PlannedBossSkill(boss.skills[0], ["A"]),
            PlannedBossSkill(boss.skills[1], ["A", "B", "C"]),
            PlannedBossSkill(boss.skills[3], ["B"]),
        ]
        actions = [
            PlayerAction("A", units[0].queue.available[0], clash_slot_index=0, armed_position=0),
            PlayerAction("B", units[1].queue.available[0], clash_slot_index=None, armed_position=0),
            PlayerAction("C", units[2].queue.available[0], clash_slot_index=None, armed_position=0),
        ]
        if via_steps:
            for _ in battle.resolve_turn_steps(slots, actions):
                pass
        else:
            battle.resolve_turn(slots, actions)
        return [u.hp for u in units] + [boss.hp]

    result_atomic = run(via_steps=False)
    result_stepped = run(via_steps=True)
    assert result_atomic == result_stepped, f"{result_atomic} != {result_stepped}"
    print("test_resolve_turn_and_resolve_turn_steps_produce_identical_final_state passed.")


def test_turn_end_step_contains_status_effect_lines():
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=random.Random(5))
    boss.burn_potency = 5
    boss.burn_count = 2

    slots = [PlannedBossSkill(boss.skills[3], ["A"])]
    actions = [PlayerAction("A", units[0].queue.available[0], clash_slot_index=None, armed_position=0)]

    steps = list(battle.resolve_turn_steps(slots, actions))
    turn_end_step = steps[-1]
    assert turn_end_step.kind == "turn_end"
    assert any("burn damage" in line for line in turn_end_step.log_lines)
    print("test_turn_end_step_contains_status_effect_lines passed.")


def test_partial_iteration_still_applies_state_up_to_that_point():
    """Stopping partway through the generator should leave state changes from
    already-yielded steps applied, with later steps not yet having happened."""
    units = make_units()
    boss = make_boss()
    battle = Battle(boss, units, rng=random.Random(7))

    slots = [
        PlannedBossSkill(boss.skills[3], ["A"]),
        PlannedBossSkill(boss.skills[3], ["B"]),
        PlannedBossSkill(boss.skills[3], ["C"]),
    ]
    actions = []  # no player actions this turn, keep it simple

    gen = battle.resolve_turn_steps(slots, actions)
    step1 = next(gen)
    assert step1.kind == "boss_unopposed"
    a_hp_after_1 = units[0].hp
    assert a_hp_after_1 < 100  # A was hit
    assert units[1].hp == 100  # B not yet hit
    assert units[2].hp == 100  # C not yet hit

    next(gen)
    assert units[1].hp < 100
    assert units[2].hp == 100  # still untouched until its own step
    print("test_partial_iteration_still_applies_state_up_to_that_point passed.")


if __name__ == "__main__":
    test_steps_cover_all_boss_slots_and_unopposed_actions()
    test_clash_step_has_roll_and_winner_info()
    test_boss_unopposed_step_participants_include_all_aoe_targets()
    test_resolve_turn_and_resolve_turn_steps_produce_identical_final_state()
    test_turn_end_step_contains_status_effect_lines()
    test_partial_iteration_still_applies_state_up_to_that_point()
    print("\nALL TURN STEP TESTS PASSED")
