from engine import PlayerUnit, SkillDef, Boss, BossSkillDef, Battle, PlayerAction, PlannedBossSkill


def make_test_unit(name, max_hp, thresholds):
    return PlayerUnit(
        name=name, max_hp=max_hp,
        skill1=SkillDef("S1", 14, 20, 25),
        skill2=SkillDef("S2", 10, 16, 50),
        stagger_thresholds=thresholds,
    )


def make_test_boss(thresholds=None):
    skills = [
        BossSkillDef("Single1", 12, 16, 30),
        BossSkillDef("Single2", 10, 14, 20),
        BossSkillDef("AoE", 10, 13, 10, hits_all=True),
        BossSkillDef("Single3", 15, 18, 15),
        BossSkillDef("Single4", 18, 20, 5),
    ]
    return Boss(name="Boss", max_hp=200, skills=skills, stagger_thresholds=thresholds or [])


def test_threshold_crossing_triggers_stagger():
    unit = make_test_unit("U", 100, [60])
    boss = make_test_boss()
    battle = Battle(boss, [unit], rng=__import__("random").Random(0))

    from engine import BattleLog
    log = BattleLog()
    battle._apply_damage(unit, 45, log)  # 100 -> 55, crosses 60
    assert unit.hp == 55
    assert unit.is_staggered is True
    assert unit.stagger_turns_left == 1
    assert 60 in unit.passed_thresholds
    print("test_threshold_crossing_triggers_stagger passed.")


def test_damage_multiplier_applies_once_staggered():
    unit = make_test_unit("U", 100, [60])
    boss = make_test_boss()
    battle = Battle(boss, [unit], rng=__import__("random").Random(0))
    from engine import BattleLog
    log = BattleLog()

    battle._apply_damage(unit, 45, log)  # triggers stagger, no multiplier on THIS hit
    assert unit.hp == 55  # not 55 - (45*0.5) extra, i.e. no multiplier applied to the triggering hit

    hp_before = unit.hp
    dmg = battle._apply_damage(unit, 10, log)  # now staggered already, should be x1.5
    assert dmg == 15
    assert unit.hp == hp_before - 15
    print("test_damage_multiplier_applies_once_staggered passed.")


def test_multiple_thresholds_dont_extend_duration():
    unit = make_test_unit("U", 100, [70, 40])
    boss = make_test_boss()
    battle = Battle(boss, [unit], rng=__import__("random").Random(0))
    from engine import BattleLog
    log = BattleLog()

    battle._apply_damage(unit, 35, log)  # 100->65, crosses 70, first stagger
    assert unit.is_staggered
    assert unit.stagger_turns_left == 1

    # simulate turns passing without healing: force turns_left down manually to confirm
    # crossing the SECOND threshold while already staggered doesn't reset turns_left
    unit.stagger_turns_left = 0
    battle._apply_damage(unit, 30, log)  # further damage crosses 40 too
    assert unit.stagger_turns_left == 0, "crossing a second threshold while staggered should not reset duration"
    assert 40 in unit.passed_thresholds
    print("test_multiple_thresholds_dont_extend_duration passed.")


def test_staggered_unit_skill_is_skipped_and_saved():
    """
    A staggered unit's committed action should not execute (no damage to boss,
    queue not advanced), effectively 'saving' the same skill for next turn.
    """
    unit = make_test_unit("U", 100, [60])
    boss = make_test_boss()
    battle = Battle(boss, [unit], rng=__import__("random").Random(0))
    unit.is_staggered = True
    unit.stagger_turns_left = 1

    queue_len_before = len(unit.queue.queue)
    boss_hp_before = boss.hp
    unit_hp_before = unit.hp

    action = PlayerAction(unit.name, unit.queue.available[0], clash_slot_index=None, armed_position=0)
    boss_slots = [PlannedBossSkill(boss.skills[2], [unit.name])]  # AoE slot, arbitrary filler, unopposed
    log = battle.resolve_turn(boss_slots, [action])

    assert unit.hp == unit_hp_before - 15  # AoE base 10 dmg x1.5 (unit already staggered) = 15
    assert boss.hp == boss_hp_before, "boss hp should be untouched since the staggered unit's skill never executed"
    assert len(unit.queue.queue) == queue_len_before, "queue should NOT advance since the skill never executed"
    print("test_staggered_unit_skill_is_skipped_and_saved passed.")


def test_stagger_duration_two_turns_then_clears():
    unit = make_test_unit("U", 200, [150])
    boss = make_test_boss()
    battle = Battle(boss, [unit], rng=__import__("random").Random(0))

    def do_turn(clash=None):
        action = PlayerAction(unit.name, unit.queue.available[0], clash_slot_index=clash, armed_position=0)
        slots = [
            PlannedBossSkill(boss.skills[0], [unit.name]),
            PlannedBossSkill(boss.skills[1], [unit.name]),
            PlannedBossSkill(boss.skills[3], [unit.name]),
        ]
        return battle.resolve_turn(slots, [action])

    # Turn 1: three boss hits of 30+20+15=65 dmg total -> hp 200->135, crosses 150 partway through.
    # By the time resolve_turn returns, turn-end processing has already ticked the duration down
    # by 1 (1 -> 0), since "staggered through the entirety of the NEXT turn" is enforced by keeping
    # is_staggered True through all of that next turn's resolution, then clearing it at ITS turn-end.
    do_turn()
    assert unit.is_staggered, "should still be staggered after turn 1 (duration covers all of turn 2 too)"
    boss_hp_before_turn1_action = boss.hp  # unit's action this turn should have been skipped too,
    # since stagger triggered mid-turn-1 from the boss's OWN hits landing before the unit's turn.

    # Turn 2: unit is staggered for the unit's ENTIRE turn 2 resolution (skill skipped/saved,
    # no damage dealt to boss), and by the end of resolve_turn's internal turn-end processing,
    # the duration naturally expires (this is what unstaggers it going into turn 3).
    boss_hp_before_turn2 = boss.hp
    do_turn()
    assert boss.hp == boss_hp_before_turn2, "unit's skill should not have executed while staggered during turn 2"
    assert not unit.is_staggered, "duration should have expired at the end of turn 2's processing"

    # Turn 3: fully unstaggered now, unit's skill should actually land on the boss.
    boss_hp_before_turn3 = boss.hp
    do_turn()
    assert boss.hp < boss_hp_before_turn3, "unit should be able to act normally again in turn 3"
    print("test_stagger_duration_two_turns_then_clears passed.")


def test_boss_stagger_blocks_boss_skills():
    unit = make_test_unit("U", 500, [])
    boss = make_test_boss(thresholds=[150])
    battle = Battle(boss, [unit], rng=__import__("random").Random(0))
    boss.is_staggered = True
    boss.stagger_turns_left = 1

    unit_hp_before = unit.hp
    action = PlayerAction(unit.name, unit.queue.available[0], clash_slot_index=None, armed_position=0)
    slots = [PlannedBossSkill(boss.skills[0], [unit.name])]
    battle.resolve_turn(slots, [action])

    assert unit.hp == unit_hp_before, "boss's skill should not have executed while boss is staggered"
    print("test_boss_stagger_blocks_boss_skills passed.")


if __name__ == "__main__":
    test_threshold_crossing_triggers_stagger()
    test_damage_multiplier_applies_once_staggered()
    test_multiple_thresholds_dont_extend_duration()
    test_staggered_unit_skill_is_skipped_and_saved()
    test_stagger_duration_two_turns_then_clears()
    test_boss_stagger_blocks_boss_skills()
    print("\nALL STAGGER TESTS PASSED")
