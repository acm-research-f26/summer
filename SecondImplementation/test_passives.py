import random
from engine import (
    PlayerUnit, SkillDef, Boss, BossSkillDef, Battle, PlayerAction, PlannedBossSkill,
    passive_roll_bonus_tremor_scorch, passive_roll_bonus_dark_flame,
    passive_roll_bonus_self_status, passive_damage_reduction_self_status,
)


def make_boss(roll=(10, 10)):
    skills = [BossSkillDef("B1", roll[0], roll[1], 20)]
    return Boss(name="Boss", max_hp=1000, skills=skills)


def test_tremor_scorch_passive_no_bonus_by_default():
    unit = PlayerUnit(name="U", max_hp=200, skill1=SkillDef("S1", 10, 10, 25),
                       skill2=SkillDef("S2", 10, 10, 25), passive_roll_bonus_fn=passive_roll_bonus_tremor_scorch)
    boss = make_boss()
    assert passive_roll_bonus_tremor_scorch(unit, boss) == 0.0
    print("test_tremor_scorch_passive_no_bonus_by_default passed.")


def test_tremor_scorch_passive_bonus_from_opponent_burn():
    unit = PlayerUnit(name="U", max_hp=200, skill1=SkillDef("S1", 10, 10, 25), skill2=SkillDef("S2", 10, 10, 25))
    boss = make_boss()
    boss.burn_potency = 15
    assert passive_roll_bonus_tremor_scorch(unit, boss) == 1.5
    boss.tremor_potency = 15
    assert passive_roll_bonus_tremor_scorch(unit, boss) == 3.0
    print("test_tremor_scorch_passive_bonus_from_opponent_burn passed.")


def test_dark_flame_passive_bonus_from_own_bullets_and_opponent_burn():
    unit = PlayerUnit(name="U", max_hp=200, skill1=SkillDef("S1", 10, 10, 25), skill2=SkillDef("S2", 10, 10, 25),
                       max_magic_bullets=7)
    boss = make_boss()
    assert passive_roll_bonus_dark_flame(unit, boss) == 0.0
    unit.magic_bullets = 5
    assert passive_roll_bonus_dark_flame(unit, boss) == 1.5
    boss.burn_potency = 15
    assert passive_roll_bonus_dark_flame(unit, boss) == 3.0
    print("test_dark_flame_passive_bonus_from_own_bullets_and_opponent_burn passed.")


def test_self_status_passive_bonus_and_damage_reduction_from_own_stacks():
    unit = PlayerUnit(name="U", max_hp=200, skill1=SkillDef("S1", 10, 10, 25), skill2=SkillDef("S2", 10, 10, 25))
    boss = make_boss()
    assert passive_roll_bonus_self_status(unit, boss) == 0.0
    assert passive_damage_reduction_self_status(unit) == 1.0

    unit.tremor_potency = 10
    assert passive_roll_bonus_self_status(unit, boss) == 1.5
    assert passive_damage_reduction_self_status(unit) == 0.8

    unit.burn_potency = 15
    assert passive_roll_bonus_self_status(unit, boss) == 3.0
    assert passive_damage_reduction_self_status(unit) == 0.6
    print("test_self_status_passive_bonus_and_damage_reduction_from_own_stacks passed.")


def test_passive_roll_bonus_actually_changes_clash_outcome():
    """End-to-end: rig a roll that would normally lose, and confirm the
    passive's bonus (applied inside _resolve_clash) flips it to a win."""
    unit = PlayerUnit(
        name="U", max_hp=200,
        skill1=SkillDef("S1", 12, 12, 25),  # always rolls exactly 12
        skill2=SkillDef("S2", 10, 10, 25),
        passive_roll_bonus_fn=passive_roll_bonus_tremor_scorch,
    )
    boss = make_boss(roll=(13, 13))  # boss always rolls exactly 13 (beats 12)
    boss.burn_potency = 15  # triggers the unit's +1.5 roll bonus -> effective 13.5 -> round to 14 (beats boss's 13)
    battle = Battle(boss, [unit], rng=random.Random(0))

    action = PlayerAction(unit.name, "skill1", clash_slot_index=0, armed_position=0)
    slot = PlannedBossSkill(boss.skills[0], [unit.name])

    boss_hp_before = boss.hp
    unit_hp_before = unit.hp
    battle.resolve_turn([slot], [action])

    assert boss.hp < boss_hp_before, "player's boosted roll should have won the clash and hit the boss"
    assert unit.hp == unit_hp_before, "player should have taken no damage, having won"
    print("test_passive_roll_bonus_actually_changes_clash_outcome passed.")


def test_passive_damage_reduction_actually_reduces_damage_taken():
    unit = PlayerUnit(
        name="U", max_hp=200,
        skill1=SkillDef("S1", 5, 5, 25),
        skill2=SkillDef("S2", 5, 5, 25),
        passive_damage_reduction_fn=passive_damage_reduction_self_status,
    )
    unit.tremor_potency = 10  # triggers 20% reduction (0.8x)
    boss = make_boss(roll=(10, 10))  # boss always beats the unit's roll of 5
    battle = Battle(boss, [unit], rng=random.Random(0))

    action = PlayerAction(unit.name, "skill1", clash_slot_index=0, armed_position=0)
    slot = PlannedBossSkill(boss.skills[0], [unit.name])

    hp_before = unit.hp
    battle.resolve_turn([slot], [action])
    dmg_taken = hp_before - unit.hp
    assert dmg_taken == 16, f"expected base 20 dmg x0.8 = 16, got {dmg_taken}"
    print("test_passive_damage_reduction_actually_reduces_damage_taken passed.")


def test_no_passive_fn_means_no_crash_and_no_bonus():
    """A unit with no passive hooks at all (the default) should behave exactly
    as before - no crash, no bonus applied."""
    unit = PlayerUnit(name="Plain", max_hp=100, skill1=SkillDef("S1", 10, 10, 25), skill2=SkillDef("S2", 10, 10, 25))
    boss = make_boss(roll=(9, 9))
    battle = Battle(boss, [unit], rng=random.Random(0))
    action = PlayerAction(unit.name, "skill1", clash_slot_index=0, armed_position=0)
    slot = PlannedBossSkill(boss.skills[0], [unit.name])
    boss_hp_before = boss.hp
    battle.resolve_turn([slot], [action])
    assert boss.hp < boss_hp_before  # unit's roll of 10 beats boss's 9, no passive needed
    print("test_no_passive_fn_means_no_crash_and_no_bonus passed.")


if __name__ == "__main__":
    test_tremor_scorch_passive_no_bonus_by_default()
    test_tremor_scorch_passive_bonus_from_opponent_burn()
    test_dark_flame_passive_bonus_from_own_bullets_and_opponent_burn()
    test_self_status_passive_bonus_and_damage_reduction_from_own_stacks()
    test_passive_roll_bonus_actually_changes_clash_outcome()
    test_passive_damage_reduction_actually_reduces_damage_taken()
    test_no_passive_fn_means_no_crash_and_no_bonus()
    print("\nALL PASSIVE TESTS PASSED")
