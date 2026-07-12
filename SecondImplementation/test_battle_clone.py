import random
from engine import PlayerUnit, SkillDef, Boss, BossSkillDef, Battle, PlayerAction


def make_units():
    return [
        PlayerUnit(name="A", max_hp=200, skill1=SkillDef("S1", 14, 20, 25), skill2=SkillDef("S2", 10, 16, 50)),
        PlayerUnit(name="B", max_hp=200, skill1=SkillDef("S1", 14, 20, 25), skill2=SkillDef("S2", 10, 16, 50)),
        PlayerUnit(name="C", max_hp=200, skill1=SkillDef("S1", 14, 20, 25), skill2=SkillDef("S2", 10, 16, 50)),
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


def test_clone_is_a_genuinely_different_object():
    battle = Battle(make_boss(), make_units(), rng=random.Random(0))
    clone = battle.clone()
    assert clone is not battle
    assert clone.boss is not battle.boss
    for name in battle.units:
        assert clone.units[name] is not battle.units[name]
    print("test_clone_is_a_genuinely_different_object passed.")


def test_mutating_the_clone_does_not_affect_the_original():
    battle = Battle(make_boss(), make_units(), rng=random.Random(0))
    original_boss_hp = battle.boss.hp
    original_unit_hp = battle.units["A"].hp

    clone = battle.clone()
    clone.boss.hp -= 500
    clone.units["A"].hp -= 100
    clone.units["A"].is_staggered = True

    assert battle.boss.hp == original_boss_hp
    assert battle.units["A"].hp == original_unit_hp
    assert battle.units["A"].is_staggered is False
    print("test_mutating_the_clone_does_not_affect_the_original passed.")


def test_resolving_a_turn_on_the_clone_does_not_touch_the_original():
    battle = Battle(make_boss(), make_units(), rng=random.Random(1))
    clone = battle.clone()

    original_state_before = ([u.hp for u in battle.units.values()], battle.boss.hp, battle.turn_number)

    # drive several turns entirely on the clone
    for _ in range(3):
        if not clone.boss.is_alive() or not clone.alive_units():
            break
        slots = clone.boss_choose_turn()
        actions = []
        for unit in clone.alive_units():
            if unit.is_staggered:
                continue
            skill_type = unit.queue.available[0]
            actions.append(PlayerAction(unit.name, skill_type, clash_slot_index=None, armed_position=0))
        clone.resolve_turn(slots, actions)

    original_state_after = ([u.hp for u in battle.units.values()], battle.boss.hp, battle.turn_number)
    assert original_state_before == original_state_after, "original battle must be completely untouched"
    assert clone.turn_number == 3
    print("test_resolving_a_turn_on_the_clone_does_not_touch_the_original passed.")


def test_default_clone_continues_the_same_random_sequence():
    """Cloning right at the start and playing identical turns on both the
    original and the clone should produce identical outcomes, since the
    clone's RNG is a faithful copy of the original's (same future sequence)."""
    def play_n_turns(battle, n):
        for _ in range(n):
            slots = battle.boss_choose_turn()
            battle.resolve_turn(slots, [])  # no player actions, pure RNG-driven boss behavior
        return [u.hp for u in battle.units.values()]

    battle_a = Battle(make_boss(), make_units(), rng=random.Random(7))
    hps_direct = play_n_turns(battle_a, 3)

    battle_b = Battle(make_boss(), make_units(), rng=random.Random(7))
    clone = battle_b.clone()
    hps_via_clone = play_n_turns(clone, 3)

    assert hps_direct == hps_via_clone, f"{hps_direct} != {hps_via_clone}"
    # and battle_b itself (the one that got cloned) should still be untouched
    assert all(u.hp == u.max_hp for u in battle_b.units.values())
    print("test_default_clone_continues_the_same_random_sequence passed.")


def test_clone_with_new_rng_uses_that_rng_going_forward():
    battle = Battle(make_boss(), make_units(), rng=random.Random(0))
    fresh_rng = random.Random(999)
    clone = battle.clone(new_rng=fresh_rng)
    assert clone.rng is fresh_rng
    assert battle.rng is not fresh_rng
    print("test_clone_with_new_rng_uses_that_rng_going_forward passed.")


def test_multiple_clones_from_the_same_state_are_mutually_independent():
    battle = Battle(make_boss(), make_units(), rng=random.Random(3))
    clone1 = battle.clone(new_rng=random.Random(101))
    clone2 = battle.clone(new_rng=random.Random(202))

    clone1.boss.hp -= 500
    clone2.boss.hp -= 100

    assert battle.boss.hp == 1500
    assert clone1.boss.hp == 1000
    assert clone2.boss.hp == 1400
    print("test_multiple_clones_from_the_same_state_are_mutually_independent passed.")


def test_clone_preserves_effect_functions_and_they_still_work():
    """A cloned unit's skill effect callables (and passive hooks) must still
    be the real, working functions - not lost or broken by the deep copy."""
    from engine import effect_tremor_scorch_skill1, passive_roll_bonus_tremor_scorch

    unit = PlayerUnit(
        name="U", max_hp=200,
        skill1=SkillDef("Tremor Jab", 14, 20, 25, effect=effect_tremor_scorch_skill1),
        skill2=SkillDef("S2", 10, 16, 50),
        passive_roll_bonus_fn=passive_roll_bonus_tremor_scorch,
    )
    battle = Battle(make_boss(), [unit], rng=random.Random(0))
    clone = battle.clone()

    cloned_unit = clone.units["U"]
    assert cloned_unit.skill1.effect is effect_tremor_scorch_skill1
    assert cloned_unit.passive_roll_bonus_fn is passive_roll_bonus_tremor_scorch

    # actually run the skill on the clone and confirm the effect fires correctly
    action = PlayerAction("U", "skill1", clash_slot_index=None, armed_position=0)
    boss_hp_before = clone.boss.hp
    clone.resolve_turn([], [action])
    assert clone.boss.hp < boss_hp_before, "cloned unit's skill effect should still deal damage"
    assert clone.boss.tremor_potency > 0, "cloned unit's skill effect should still apply tremor"
    print("test_clone_preserves_effect_functions_and_they_still_work passed.")


def test_clone_mid_battle_preserves_complex_state():
    """Clone AFTER some real state has accumulated (stagger, status effects,
    a non-zero turn number) - not just a freshly-constructed battle - to make
    sure nothing subtle gets lost for a more "in-progress" snapshot."""
    battle = Battle(make_boss(), make_units(), rng=random.Random(2))
    battle.turn_number = 5
    battle.boss.tremor_potency = 12
    battle.boss.tremor_count = 4
    battle.boss.is_staggered = True
    battle.boss.stagger_turns_left = 1
    battle.units["A"].burn_potency = 8
    battle.units["A"].burn_count = 3
    battle.units["A"].hp = 42

    clone = battle.clone()

    assert clone.turn_number == 5
    assert clone.boss.tremor_potency == 12
    assert clone.boss.tremor_count == 4
    assert clone.boss.is_staggered is True
    assert clone.boss.stagger_turns_left == 1
    assert clone.units["A"].burn_potency == 8
    assert clone.units["A"].burn_count == 3
    assert clone.units["A"].hp == 42

    # and it's still independent - mutating the clone doesn't touch the original
    clone.boss.tremor_potency = 999
    assert battle.boss.tremor_potency == 12
    print("test_clone_mid_battle_preserves_complex_state passed.")


def test_clone_skill_queue_state_is_independent():
    """The skill queue (which of a unit's skills are currently pickable) must
    also be an independent copy, not shared between original and clone."""
    battle = Battle(make_boss(), make_units(), rng=random.Random(4))
    clone = battle.clone()

    original_queue_before = list(battle.units["A"].queue.queue)
    clone.units["A"].queue.pick(clone.units["A"].queue.available[0], position=0)

    assert battle.units["A"].queue.queue == original_queue_before, "original unit's queue must be untouched"
    assert clone.units["A"].queue.queue != original_queue_before, "clone's queue should have advanced"
    print("test_clone_skill_queue_state_is_independent passed.")


if __name__ == "__main__":
    test_clone_is_a_genuinely_different_object()
    test_mutating_the_clone_does_not_affect_the_original()
    test_resolving_a_turn_on_the_clone_does_not_touch_the_original()
    test_default_clone_continues_the_same_random_sequence()
    test_clone_with_new_rng_uses_that_rng_going_forward()
    test_multiple_clones_from_the_same_state_are_mutually_independent()
    test_clone_preserves_effect_functions_and_they_still_work()
    test_clone_mid_battle_preserves_complex_state()
    test_clone_skill_queue_state_is_independent()
    print("\nALL BATTLE CLONE TESTS PASSED")
