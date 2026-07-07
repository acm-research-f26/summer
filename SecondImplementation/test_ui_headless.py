import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import random
from game_ui import GameUI


def test_basic_flow():
    game = GameUI()
    game.draw()
    assert not game._all_actions_ready()

    for unit in game.units:
        game._arm(unit.name, 0)
        game._commit_action(clash_slot_index=None)
        assert game.player_actions[unit.name].clash_slot_index is None
        assert game.player_actions[unit.name].armed_position == 0

    assert game._all_actions_ready()
    boss_hp_before = game.boss.hp
    turn_before = game.battle.turn_number
    game._start_turn()
    assert game.battle.turn_number == turn_before + 1
    assert game.boss.hp <= boss_hp_before
    assert game.player_actions == {}
    game.draw()
    print("test_basic_flow passed. boss hp:", game.boss.hp)


def test_cancel_action():
    game = GameUI()
    game.draw()
    unit = game.units[0]
    game._arm(unit.name, 0)
    game._commit_action(clash_slot_index=None)
    assert unit.name in game.player_actions
    game._cancel_action(unit.name)
    assert unit.name not in game.player_actions
    assert game.armed is None
    game.draw()
    positions = [
        c.payload[1][1] for c in game.clickables
        if c.payload[0] == "arm_skill" and c.payload[1][0] == unit.name
    ]
    assert set(positions) == {0, 1}
    print("test_cancel_action passed.")


def test_duplicate_skill_type_independent_arming():
    game = GameUI()
    unit = game.units[0]
    unit.queue.queue[0] = 'skill1'
    unit.queue.queue[1] = 'skill1'
    game.draw()
    game._arm(unit.name, 1)
    assert (game.armed == (unit.name, 0)) is False
    assert (game.armed == (unit.name, 1)) is True
    queue_len_before = len(unit.queue.queue)
    game._commit_action(clash_slot_index=None)
    action = game.player_actions[unit.name]
    unit.queue.pick(action.skill_type, position=action.armed_position)
    assert len(unit.queue.queue) == queue_len_before - 2
    print("test_duplicate_skill_type_independent_arming passed.")


def test_unopposed_button_click_path():
    game = GameUI()
    game.draw()
    unit = game.units[0]
    game._arm(unit.name, 0)
    game.draw()
    btn = next(c for c in game.clickables if c.payload == ("attack_unopposed", unit.name))
    game.handle_click(btn.rect.center)
    assert unit.name in game.player_actions
    assert game.player_actions[unit.name].clash_slot_index is None
    print("test_unopposed_button_click_path passed.")


def test_manual_boss_cycle_before_claim():
    game = GameUI()
    game.manual_boss_mode = True
    game.draw()
    slot0_before = game.boss_slots[0].skill_def.name
    game._cycle_boss_skill(0)
    assert game.boss_slots[0].skill_def.name != slot0_before
    print("test_manual_boss_cycle_before_claim passed.")


def test_manual_boss_cycle_locked_after_claim():
    """Once a slot is claimed by a clash, cycling its skill/target must be a no-op."""
    game = GameUI()
    game.manual_boss_mode = True
    game.draw()

    unit = game.units[0]
    game._arm(unit.name, 0)
    game._commit_action(clash_slot_index=0)
    assert 0 in game._claimed_slot_indices()

    skill_before = game.boss_slots[0].skill_def.name
    target_before = list(game.boss_slots[0].target_names)
    game._cycle_boss_skill(0)
    game._cycle_boss_target(0)
    assert game.boss_slots[0].skill_def.name == skill_before, "claimed slot's skill should be locked"
    assert game.boss_slots[0].target_names == target_before, "claimed slot's target should be locked"

    # also confirm the UI doesn't even offer the cycle click regions for a claimed slot
    game.draw()
    cycle_clicks = [c for c in game.clickables if c.payload[0] in ("cycle_boss_skill", "cycle_boss_target") and c.payload[1] == 0]
    assert cycle_clicks == [], "claimed slot should not expose cycle clickables"
    print("test_manual_boss_cycle_locked_after_claim passed.")


def test_retarget_to_clasher_on_boss_win():
    """
    Force a deterministic RNG where the boss always wins clashes, put a
    single-target boss skill on a unit that ISN'T its original random target,
    have a different unit clash it, and confirm damage lands on the clasher,
    not the original target.
    """
    game = GameUI()
    game.rng = random.Random(1)
    game.battle.rng = game.rng

    from engine import PlannedBossSkill
    # find a non-hits_all boss skill def
    single_target_def = next(s for s in game.boss.skills if not s.hits_all)
    other_single_target_def = next(
        s for s in game.boss.skills if not s.hits_all and s is not single_target_def
    )

    original_target = game.units[1].name  # DarkFlameInflictor
    clasher = game.units[0]                # TremorScorch
    third_unit = game.units[2].name        # BurnTremorInflictor

    # Slots 1 and 2 are pinned to non-hits_all skills aimed only at the third
    # unit, so they can't accidentally damage original_target and confound the check.
    game.boss_slots = [
        PlannedBossSkill(single_target_def, [original_target]),
        PlannedBossSkill(other_single_target_def, [third_unit]),
        PlannedBossSkill(other_single_target_def, [third_unit]),
    ]

    game._arm(clasher.name, 0)
    game._commit_action(clash_slot_index=0)
    for u in game.units[1:]:
        game._arm(u.name, 0)
        game._commit_action(clash_slot_index=None)

    hp_before_original_target = game.units[1].hp
    hp_before_clasher = clasher.hp

    # Force boss to always win by rigging rolls: monkeypatch _roll to make boss roll high, player roll low
    original_roll = game.battle._roll
    def rigged_roll(lo, hi):
        # crude: if this is being called for the boss skill's own range, return hi; else lo
        return hi if (lo, hi) == (single_target_def.roll_lo, single_target_def.roll_hi) else lo
    game.battle._roll = rigged_roll

    game._start_turn()
    game.battle._roll = original_roll

    took_damage_original = game.units[1].hp < hp_before_original_target
    took_damage_clasher = clasher.hp < hp_before_clasher

    assert not took_damage_original, "boss should NOT have hit the original random target once clashed"
    assert took_damage_clasher, "boss should have redirected onto the clasher and hit them instead"
    print("test_retarget_to_clasher_on_boss_win passed.")


def test_tooltip_hover_regions_present():
    game = GameUI()
    game.draw()
    # boss passive/info tooltip present
    assert any(hr.payload == game.boss.passive_description for hr in game.hover_regions)
    # each unit's passive tooltip present
    for unit in game.units:
        assert any(hr.payload == unit.passive_description for hr in game.hover_regions)
    # skill description tooltips present (at least skill1 desc of first unit)
    unit0 = game.units[0]
    assert any(hr.payload == unit0.skill1.description for hr in game.hover_regions)
    assert any(hr.payload == unit0.skill2.description for hr in game.hover_regions)
    # boss slot skill descriptions present
    for slot in game.boss_slots:
        assert any(hr.payload == slot.skill_def.description for hr in game.hover_regions)
    print("test_tooltip_hover_regions_present passed.")


def test_dead_unit_excluded_from_clickables():
    game = GameUI()
    game.units[1].hp = 0
    game.draw()
    kinds = [
        c.payload for c in game.clickables
        if c.payload[0] == "arm_skill" and c.payload[1][0] == game.units[1].name
    ]
    assert kinds == []
    print("test_dead_unit_excluded_from_clickables passed.")


def test_no_boss_portrait_clickable_anymore():
    """The old 'click boss portrait to go unopposed' shortcut was replaced by
    the explicit per-unit ATTACK UNOPPOSED button; portrait is hover-only now."""
    game = GameUI()
    game.draw()
    assert not any(c.payload[0] == "boss_portrait" for c in game.clickables)
    print("test_no_boss_portrait_clickable_anymore passed.")


def test_staggered_unit_excluded_from_action_requirement():
    game = GameUI()
    game.draw()
    staggered_unit = game.units[1]
    staggered_unit.is_staggered = True
    staggered_unit.stagger_turns_left = 1

    for unit in game.units:
        if unit is staggered_unit:
            continue
        game._arm(unit.name, 0)
        game._commit_action(clash_slot_index=None)

    assert game._all_actions_ready(), "staggered unit should not block readiness"
    game.draw()
    # staggered unit should show no arm_skill clickables and no unopposed button
    kinds = [c.payload for c in game.clickables if c.payload[0] in ("arm_skill", "attack_unopposed")
             and (c.payload[1][0] if c.payload[0] == "arm_skill" else c.payload[1]) == staggered_unit.name]
    assert kinds == [], "staggered unit should not expose action clickables"
    print("test_staggered_unit_excluded_from_action_requirement passed.")


def test_stagger_tick_marks_and_ring_present_when_staggered():
    game = GameUI()
    game.units[0].is_staggered = True
    game.units[0].stagger_turns_left = 1
    game.boss.is_staggered = True
    game.boss.stagger_turns_left = 1
    game.draw()  # should not crash while drawing staggered rings/labels
    print("test_stagger_tick_marks_and_ring_present_when_staggered passed.")


def test_badge_hidden_when_count_zero_even_with_lingering_potency():
    game = GameUI()
    unit = game.units[0]
    # simulate a lingering-potency edge case (shouldn't normally happen given the
    # engine's own reset-on-decay, but the UI should be defensive about it anyway)
    unit.tremor_potency = 5
    unit.tremor_count = 0
    unit.burn_potency = 7
    unit.burn_count = 0
    game.draw()
    # no hover region should mention tremor/burn for this unit's badge area if count is 0
    # (we check indirectly: no badge-sized hover rect centered at the badge row for this unit
    # carries tremor/burn tooltip text)
    badge_tooltips = [
        hr.payload for hr in game.hover_regions
        if "TREMOR" in hr.payload or "BURN (" in hr.payload
    ]
    assert badge_tooltips == [], "no tremor/burn badge tooltip should exist when count is 0"
    print("test_badge_hidden_when_count_zero_even_with_lingering_potency passed.")


def test_badge_tooltips_present_when_active():
    game = GameUI()
    unit = game.units[0]
    unit.tremor_potency = 3
    unit.tremor_count = 2
    game.boss.burn_potency = 5
    game.boss.burn_count = 2
    game.boss.dark_flame_count = 1
    game.draw()

    assert any("TREMOR" in hr.payload for hr in game.hover_regions)
    assert any("BURN (" in hr.payload for hr in game.hover_regions)
    assert any("DARK FLAME" in hr.payload for hr in game.hover_regions)
    print("test_badge_tooltips_present_when_active passed.")


def test_scorch_vs_normal_tremor_tooltip_differ():
    game = GameUI()
    unit = game.units[0]
    unit.tremor_potency = 3
    unit.tremor_count = 2
    unit.tremor_type = "normal"
    game.draw()
    normal_tooltip = next(hr.payload for hr in game.hover_regions if "TREMOR" in hr.payload)

    unit.tremor_type = "scorch"
    game.draw()
    scorch_tooltip = next(hr.payload for hr in game.hover_regions if "TREMOR" in hr.payload)

    assert normal_tooltip != scorch_tooltip
    assert "SCORCH" in scorch_tooltip
    print("test_scorch_vs_normal_tremor_tooltip_differ passed.")


if __name__ == "__main__":
    test_basic_flow()
    test_cancel_action()
    test_duplicate_skill_type_independent_arming()
    test_unopposed_button_click_path()
    test_manual_boss_cycle_before_claim()
    test_manual_boss_cycle_locked_after_claim()
    test_retarget_to_clasher_on_boss_win()
    test_tooltip_hover_regions_present()
    test_dead_unit_excluded_from_clickables()
    test_no_boss_portrait_clickable_anymore()
    test_staggered_unit_excluded_from_action_requirement()
    test_stagger_tick_marks_and_ring_present_when_staggered()
    test_badge_hidden_when_count_zero_even_with_lingering_potency()
    test_badge_tooltips_present_when_active()
    test_scorch_vs_normal_tremor_tooltip_differ()
    print("\nALL TESTS PASSED")
