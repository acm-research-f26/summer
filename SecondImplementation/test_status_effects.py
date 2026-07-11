import random
from engine import (
    PlayerUnit, SkillDef, Boss, BossSkillDef, Battle, PlayerAction, PlannedBossSkill, BattleLog,
    effect_tremor_scorch_skill1, effect_tremor_scorch_skill2,
    effect_dark_flame_skill1, effect_dark_flame_skill2,
    effect_self_status_skill1, effect_self_status_skill2,
    effect_boss_tremor_slam, effect_boss_burn_wave, effect_boss_amplitude_cascade,
)


def make_tremor_scorch():
    return PlayerUnit(
        name="Thumb East Capo III", max_hp=130,
        skill1=SkillDef("Tremor Jab", 14, 20, 25, effect=effect_tremor_scorch_skill1),
        skill2=SkillDef("Tremor Burst Strike", 10, 16, 50, effect=effect_tremor_scorch_skill2),
        stagger_thresholds=[65],
    )


def make_dark_flame():
    return PlayerUnit(
        name="Lobotomy EGO: Magic Bullet", max_hp=150,
        skill1=SkillDef("Flame Tag", 14, 20, 25, effect=effect_dark_flame_skill1),
        skill2=SkillDef("Dark Flame Surge", 10, 16, 50, effect=effect_dark_flame_skill2),
        stagger_thresholds=[110, 70, 30],
        max_magic_bullets=7,
    )


def make_self_status():
    return PlayerUnit(
        name="You Branch Adept", max_hp=100,
        skill1=SkillDef("Shared Tremor", 14, 20, 25, effect=effect_self_status_skill1),
        skill2=SkillDef("Shared Burn", 10, 16, 50, effect=effect_self_status_skill2),
        stagger_thresholds=[60, 30],
    )


def make_boss(thresholds=None):
    skills = [
        BossSkillDef("Tremor Slam", 12, 16, 25, effect=effect_boss_tremor_slam),
        BossSkillDef("Burn Wave", 10, 13, 10, hits_all=True, effect=effect_boss_burn_wave),
        BossSkillDef("Amplitude Cascade", 18, 20, 40, hits_all=True, effect=effect_boss_amplitude_cascade),
        BossSkillDef("Filler1", 15, 18, 5),
        BossSkillDef("Filler2", 10, 12, 5),
    ]
    return Boss(name="Boss", max_hp=1500, skills=skills, stagger_thresholds=thresholds or [])


def test_tremor_scorch_skill1_applies_tremor_and_conditional_burst():
    unit = make_tremor_scorch()
    boss = make_boss(thresholds=[1400])  # low enough that flat skill damage alone won't cross it
    battle = Battle(boss, [unit], rng=random.Random(0))
    log = BattleLog()

    effect_tremor_scorch_skill1(battle, unit, boss, log, unit.skill1)
    assert boss.tremor_potency == 3
    assert boss.tremor_count == 2
    assert boss.hp == 1500 - 25
    # count (2) is not > 3, so no burst should have happened yet -> threshold unchanged
    assert 1400 in boss.stagger_thresholds

    # applying again: potency 6, count 4 -> now count(4) > 3 -> should burst, raising the threshold
    effect_tremor_scorch_skill1(battle, unit, boss, log, unit.skill1)
    assert boss.tremor_potency == 6
    assert boss.tremor_count == 3  # 4 inflicted, then burst reduces by 1 -> 3
    assert 1400 not in boss.stagger_thresholds  # raised by 6 -> now 1406
    assert 1406 in boss.stagger_thresholds
    print("test_tremor_scorch_skill1_applies_tremor_and_conditional_burst passed.")


def test_tremor_scorch_skill2_converts_and_double_bursts():
    unit = make_tremor_scorch()
    boss = make_boss()
    boss.tremor_potency = 10
    boss.tremor_count = 5
    battle = Battle(boss, [unit], rng=random.Random(0))
    log = BattleLog()

    effect_tremor_scorch_skill2(battle, unit, boss, log, unit.skill2)
    assert boss.tremor_potency == 12  # +2
    assert boss.tremor_type == "scorch"
    # skill2 adds 0 count; two bursts each reduce count by 1: 5 -> 4 -> 3
    assert boss.tremor_count == 3
    # both bursts should have dealt scorch physical damage (potency 12 + burn potency 0 = 12 each)
    expected_hp = 1500 - 50 - 12 - 12  # base dmg 50, then two scorch bursts of 12 each
    assert boss.hp == expected_hp, f"expected {expected_hp}, got {boss.hp}"
    print("test_tremor_scorch_skill2_converts_and_double_bursts passed.")


def test_dark_flame_skill1_uses_current_bullets_then_gains():
    unit = make_dark_flame()
    unit.magic_bullets = 3
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    log = BattleLog()

    effect_dark_flame_skill1(battle, unit, boss, log, unit.skill1)
    assert boss.burn_potency == 3, "burn should equal bullets BEFORE this skill's own gain"
    assert boss.dark_flame_count == 1
    assert unit.magic_bullets == 5  # 3 + 2
    assert boss.hp == 1500 - 25
    print("test_dark_flame_skill1_uses_current_bullets_then_gains passed.")


def test_dark_flame_skill1_caps_at_max():
    unit = make_dark_flame()
    unit.magic_bullets = 6
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    log = BattleLog()

    effect_dark_flame_skill1(battle, unit, boss, log, unit.skill1)
    assert unit.magic_bullets == 7, "should cap at max_magic_bullets, not go to 8"
    print("test_dark_flame_skill1_caps_at_max passed.")


def test_dark_flame_skill2_gains_then_uses_updated_bullets():
    unit = make_dark_flame()
    unit.magic_bullets = 2
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    log = BattleLog()

    effect_dark_flame_skill2(battle, unit, boss, log, unit.skill2)
    assert unit.magic_bullets == 3  # gained 1 first
    assert boss.dark_flame_count == 3, "dark flame should use the UPDATED bullet count (3), not the old (2)"
    print("test_dark_flame_skill2_gains_then_uses_updated_bullets passed.")


def test_self_status_skill1_applies_to_both_self_and_target():
    unit = make_self_status()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    log = BattleLog()

    effect_self_status_skill1(battle, unit, boss, log, unit.skill1)
    assert boss.tremor_potency == 16 and boss.tremor_count == 8
    assert unit.tremor_potency == 5 and unit.tremor_count == 3
    print("test_self_status_skill1_applies_to_both_self_and_target passed.")


def test_self_status_skill2_nonlethal_floor_at_1():
    unit = make_self_status()
    unit.hp = 3  # low HP so the burn tick would normally kill it
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    log = BattleLog()

    effect_self_status_skill2(battle, unit, boss, log, unit.skill2)
    assert unit.burn_nonlethal is True
    assert boss.burn_nonlethal is True
    assert unit.burn_potency == 10 and unit.burn_count == 5
    assert boss.burn_potency == 10 and boss.burn_count == 5

    # now simulate the turn-end burn tick and confirm it floors at 1, not below
    battle._process_status_effects_turn_end(log)
    assert unit.hp == 1, f"expected floor at 1, got {unit.hp}"
    print("test_self_status_skill2_nonlethal_floor_at_1 passed.")


def test_self_status_skill2_base_damage_includes_own_burn_potency():
    unit = make_self_status()
    unit.burn_potency = 20  # pre-existing burn potency on self
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    log = BattleLog()

    boss_hp_before = boss.hp
    effect_self_status_skill2(battle, unit, boss, log, unit.skill2)
    dmg_dealt = boss_hp_before - boss.hp
    assert dmg_dealt == 70, f"expected base 50+20=70 dmg, got {dmg_dealt}"
    print("test_self_status_skill2_base_damage_includes_own_burn_potency passed.")


def test_burn_tick_at_turn_end_normal():
    unit = make_tremor_scorch()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    boss.burn_potency = 8
    boss.burn_count = 3
    hp_before = boss.hp
    log = BattleLog()
    battle._process_status_effects_turn_end(log)
    assert boss.hp == hp_before - 8
    assert boss.burn_count == 2
    print("test_burn_tick_at_turn_end_normal passed.")


def test_dark_flame_burn_interaction_at_turn_end():
    """If dark_flame_count > 0 at turn end, damage = dark_flame_count * burn_potency instead
    of just burn_potency, and dark_flame is consumed."""
    unit = make_tremor_scorch()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    boss.burn_potency = 5
    boss.burn_count = 2
    boss.dark_flame_count = 3
    hp_before = boss.hp
    log = BattleLog()
    battle._process_status_effects_turn_end(log)
    assert boss.hp == hp_before - 15, "should take dark_flame_count(3) * burn_potency(5) = 15"
    assert boss.dark_flame_count == 0
    assert boss.burn_count == 1
    print("test_dark_flame_burn_interaction_at_turn_end passed.")


def test_tremor_count_decays_at_turn_end():
    unit = make_tremor_scorch()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    boss.tremor_count = 4
    log = BattleLog()
    battle._process_status_effects_turn_end(log)
    assert boss.tremor_count == 3
    print("test_tremor_count_decays_at_turn_end passed.")


def test_amplitude_conversion_expires_after_two_turns():
    unit = make_tremor_scorch()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    log = BattleLog()

    battle.turn_number = 5
    battle._trigger_amplitude_conversion(boss, log)
    assert boss.tremor_type == "scorch"
    assert boss.tremor_scorch_expire_turn == 7

    battle.turn_number = 6
    battle._process_status_effects_turn_end(log)
    assert boss.tremor_type == "scorch", "should still be scorch (6 < 7)"

    battle.turn_number = 7
    battle._process_status_effects_turn_end(log)
    assert boss.tremor_type == "normal", "should revert once turn_number reaches expire_turn"
    print("test_amplitude_conversion_expires_after_two_turns passed.")


def test_reapplying_amplitude_conversion_resets_deadline():
    unit = make_tremor_scorch()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    log = BattleLog()

    battle.turn_number = 5
    battle._trigger_amplitude_conversion(boss, log)
    assert boss.tremor_scorch_expire_turn == 7

    battle.turn_number = 6
    battle._trigger_amplitude_conversion(boss, log)  # reapplied while still scorch
    assert boss.tremor_scorch_expire_turn == 8, "deadline should push back to 6+2=8"
    print("test_reapplying_amplitude_conversion_resets_deadline passed.")


def test_full_turn_integration_with_status_effects():
    """Smoke test: a full resolve_turn() with real skills produces sane state changes."""
    unit = make_dark_flame()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))

    action = PlayerAction(unit.name, unit.queue.available[0], clash_slot_index=None, armed_position=0)
    slots = [PlannedBossSkill(boss.skills[1], [unit.name])]  # Burn Wave (AoE)
    log = battle.resolve_turn(slots, [action])
    assert isinstance(log.dump(), str)
    assert unit.hp <= 150  # took some damage or stayed same
    print("test_full_turn_integration_with_status_effects passed.")


def test_potency_clears_when_count_decays_to_zero_via_turn_end():
    unit = make_tremor_scorch()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    boss.burn_potency = 8
    boss.burn_count = 1
    boss.tremor_potency = 4
    boss.tremor_count = 1
    log = BattleLog()

    battle._process_status_effects_turn_end(log)
    assert boss.burn_count == 0
    assert boss.burn_potency == 0, "burn potency should clear once count naturally hits 0"
    assert boss.tremor_count == 0
    assert boss.tremor_potency == 0, "tremor potency should clear once count naturally hits 0"
    print("test_potency_clears_when_count_decays_to_zero_via_turn_end passed.")


def test_potency_clears_when_tremor_count_hits_zero_via_burst():
    unit = make_tremor_scorch()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    boss.tremor_potency = 7
    boss.tremor_count = 1
    log = BattleLog()

    battle._tremor_burst(boss, log)
    assert boss.tremor_count == 0
    assert boss.tremor_potency == 0, "tremor potency should clear once burst brings count to 0"
    print("test_potency_clears_when_tremor_count_hits_zero_via_burst passed.")


def test_scorch_burst_still_uses_correct_potency_even_as_count_hits_zero():
    """The burst that brings count to 0 should still deal scorch damage using
    THAT burst's potency, not the zeroed-out post-reset value."""
    unit = make_tremor_scorch()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    boss.tremor_potency = 10
    boss.tremor_count = 1
    boss.tremor_type = "scorch"
    boss.burn_potency = 5
    hp_before = boss.hp
    log = BattleLog()

    battle._tremor_burst(boss, log)
    assert boss.tremor_count == 0
    assert boss.tremor_potency == 0
    assert boss.hp == hp_before - 15, "scorch damage should still be 10+5=15, using pre-reset potency"
    print("test_scorch_burst_still_uses_correct_potency_even_as_count_hits_zero passed.")


def test_tremor_count_management_burst_twice_plus_turn_end():
    """
    End-to-end proof of the documented rule: 'if you tremor bursted twice and
    then reach end of the turn, the tremor count would have decreased by 3'
    (one decrement per burst, plus one more at turn end) - exercised through
    the real Tremor Burst Strike skill (which bursts twice) inside a full
    resolve_turn(), not just the isolated helper methods.
    """
    unit = make_tremor_scorch()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))

    # give the target a healthy starting tremor count so we can watch it drop by exactly 3
    boss.tremor_count = 10
    boss.tremor_potency = 5

    position = unit.queue.available.index("skill2")
    action = PlayerAction(unit.name, "skill2", clash_slot_index=None, armed_position=position)
    # keep the boss's own 3 slots harmless/unrelated so only our skill's bursts matter
    filler = BossSkillDef("Filler", 10, 12, 1)
    slots = [PlannedBossSkill(filler, [unit.name]) for _ in range(3)]

    battle.resolve_turn(slots, [action])

    # Tremor Burst Strike (effect_tremor_scorch_skill2) adds +2 potency/+0 count,
    # then bursts twice (-1 count each = -2), then turn-end ticks count down
    # once more (-1) -> total count change from the skill's two bursts + turn-end = -3
    # starting count 10 -> after skill's own +0 count change still 10 -> burst -> 9
    # -> burst -> 8 -> turn-end -> 7
    assert boss.tremor_count == 7, f"expected count to drop by 3 (2 bursts + turn-end), got {boss.tremor_count}"
    print("test_tremor_count_management_burst_twice_plus_turn_end passed.")


def test_burn_count_decrements_exactly_once_at_turn_end():
    unit = make_tremor_scorch()
    boss = make_boss()
    battle = Battle(boss, [unit], rng=random.Random(0))
    boss.burn_potency = 5
    boss.burn_count = 4
    log = BattleLog()
    battle._process_status_effects_turn_end(log)
    assert boss.burn_count == 3, "burn count should decrement by exactly 1 at turn end"
    # calling it again (simulating a second, separate turn end) should decrement once more
    battle._process_status_effects_turn_end(log)
    assert boss.burn_count == 2
    print("test_burn_count_decrements_exactly_once_at_turn_end passed.")


if __name__ == "__main__":
    test_tremor_scorch_skill1_applies_tremor_and_conditional_burst()
    test_tremor_scorch_skill2_converts_and_double_bursts()
    test_dark_flame_skill1_uses_current_bullets_then_gains()
    test_dark_flame_skill1_caps_at_max()
    test_dark_flame_skill2_gains_then_uses_updated_bullets()
    test_self_status_skill1_applies_to_both_self_and_target()
    test_self_status_skill2_nonlethal_floor_at_1()
    test_self_status_skill2_base_damage_includes_own_burn_potency()
    test_burn_tick_at_turn_end_normal()
    test_dark_flame_burn_interaction_at_turn_end()
    test_tremor_count_decays_at_turn_end()
    test_amplitude_conversion_expires_after_two_turns()
    test_reapplying_amplitude_conversion_resets_deadline()
    test_full_turn_integration_with_status_effects()
    test_potency_clears_when_count_decays_to_zero_via_turn_end()
    test_potency_clears_when_tremor_count_hits_zero_via_burst()
    test_scorch_burst_still_uses_correct_potency_even_as_count_hits_zero()
    test_tremor_count_management_burst_twice_plus_turn_end()
    test_burn_count_decrements_exactly_once_at_turn_end()
    print("\nALL STATUS EFFECT TESTS PASSED")
