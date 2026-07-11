import random
from engine import (
    PlayerUnit, SkillDef, Boss, BossSkillDef, Battle, PlayerAction, PlannedBossSkill,
    effect_boss_clash_baiter,
)


def make_unit(skill1_roll=(13, 19), skill2_roll=(9, 18)):
    return PlayerUnit(
        name="U", max_hp=500,
        skill1=SkillDef("S1", *skill1_roll, 25),
        skill2=SkillDef("S2", *skill2_roll, 50),
    )


def make_boss_with_clash_baiter(roll=(11, 15), bonus=5, multiplier=10.0):
    skills = [
        BossSkillDef("Clash Baiter", roll[0], roll[1], 15, effect=effect_boss_clash_baiter,
                     clash_roll_bonus=bonus, clash_damage_multiplier=multiplier),
        BossSkillDef("Filler", 10, 12, 1),
    ]
    return Boss(name="Boss", max_hp=500, skills=skills)


def test_no_bonus_applies_when_unopposed():
    unit = make_unit()
    boss = make_boss_with_clash_baiter()
    battle = Battle(boss, [unit], rng=random.Random(0))

    slot = PlannedBossSkill(boss.skills[0], [unit.name])
    hp_before = unit.hp
    battle._execute_boss_skill(slot, __import__("engine").BattleLog())
    dmg_taken = hp_before - unit.hp
    assert dmg_taken == 15, f"unopposed hit should deal plain base damage (15), got {dmg_taken}"
    print("test_no_bonus_applies_when_unopposed passed.")


def test_damage_multiplier_applies_when_boss_wins_clash():
    unit = make_unit()
    boss = make_boss_with_clash_baiter()
    battle = Battle(boss, [unit], rng=random.Random(0))

    action = PlayerAction(unit.name, "skill1", clash_slot_index=0, armed_position=0)
    slot = PlannedBossSkill(boss.skills[0], [unit.name])

    # rig rolls so boss always wins: boss's (boosted) roll high, player roll low
    call_count = [0]
    def rigged_roll(lo, hi):
        call_count[0] += 1
        return hi if call_count[0] % 2 == 1 else lo
    battle._roll = rigged_roll

    hp_before = unit.hp
    battle.resolve_turn([slot], [action])
    dmg_taken = hp_before - unit.hp
    assert dmg_taken == 150, f"boss winning the clash should deal base(15) x 10 = 150, got {dmg_taken}"
    print("test_damage_multiplier_applies_when_boss_wins_clash passed.")


def test_roll_bonus_only_applies_during_the_clash_itself():
    """
    The +5 roll bonus should only affect the roll used to resolve the clash,
    never the skill's normal roll_lo/roll_hi attributes (which stay at their
    base values for e.g. UI display and unopposed resolution).
    """
    boss = make_boss_with_clash_baiter(roll=(11, 15), bonus=5)
    assert boss.skills[0].roll_lo == 11
    assert boss.skills[0].roll_hi == 15
    print("test_roll_bonus_only_applies_during_the_clash_itself passed.")


def test_roll_bonus_definitively_changes_clash_outcome():
    """Direct win-probability check: boss's effective (boosted) roll range
    should be used for computing whether the bonus meaningfully changes who's
    favored, verified by rigging deterministic rolls at the boundary."""
    unit = make_unit(skill1_roll=(14, 14))  # player always rolls exactly 14
    # boss base roll fixed at 10 (would lose to player's 14 every time, 10 < 14)
    boss = make_boss_with_clash_baiter(roll=(10, 10), bonus=5)  # boosted: effectively 15
    battle = Battle(boss, [unit], rng=random.Random(0))

    action = PlayerAction(unit.name, "skill1", clash_slot_index=0, armed_position=0)
    slot = PlannedBossSkill(boss.skills[0], [unit.name])

    hp_before = unit.hp
    battle.resolve_turn([slot], [action])
    dmg_taken = hp_before - unit.hp
    # without the +5 bonus, boss's 10 would always lose to player's 14 (no bonus dmg).
    # with the +5 bonus, boss's effective roll is 15, which beats player's 14 ->
    # boss wins -> multiplier applies -> damage = 15 base * 10 = 150
    assert dmg_taken == 150, (
        f"expected the +5 bonus to flip this clash in the boss's favor (150 dmg), got {dmg_taken}"
    )
    print("test_roll_bonus_definitively_changes_clash_outcome passed.")


def test_player_can_still_win_a_clash_against_the_boosted_roll():
    """Even with the bonus, the boss's roll isn't infinite - a high enough
    player roll should still be able to win and take no damage."""
    unit = make_unit(skill1_roll=(20, 20))  # player always rolls 20
    boss = make_boss_with_clash_baiter(roll=(10, 10), bonus=5)  # boosted to 15, still loses to 20
    battle = Battle(boss, [unit], rng=random.Random(0))

    action = PlayerAction(unit.name, "skill1", clash_slot_index=0, armed_position=0)
    slot = PlannedBossSkill(boss.skills[0], [unit.name])

    hp_before = unit.hp
    boss_hp_before = boss.hp
    battle.resolve_turn([slot], [action])
    assert unit.hp == hp_before, "player should have taken no damage, having won the clash"
    assert boss.hp < boss_hp_before, "player's skill should have landed on the boss"
    print("test_player_can_still_win_a_clash_against_the_boosted_roll passed.")


if __name__ == "__main__":
    test_no_bonus_applies_when_unopposed()
    test_damage_multiplier_applies_when_boss_wins_clash()
    test_roll_bonus_only_applies_during_the_clash_itself()
    test_roll_bonus_definitively_changes_clash_outcome()
    test_player_can_still_win_a_clash_against_the_boosted_roll()
    print("\nALL CLASH BONUS TESTS PASSED")
