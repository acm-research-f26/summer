import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from game_ui import GameUI


def test_basic_flow():
    game = GameUI()
    game.draw()

    assert not game._all_actions_ready()

    for unit in game.units:
        game._arm(unit.name, 0)  # arm the BOTTOM slot
        assert game.armed == (unit.name, 0)
        game._commit_action(clash_slot_index=None)
        assert unit.name in game.player_actions
        assert game.player_actions[unit.name].clash_slot_index is None
        assert game.player_actions[unit.name].armed_position == 0

    assert game._all_actions_ready()

    boss_hp_before = game.boss.hp
    turn_before = game.battle.turn_number
    game._start_turn()
    assert game.battle.turn_number == turn_before + 1
    assert game.boss.hp <= boss_hp_before
    assert game.player_actions == {}
    assert game.armed is None
    game.draw()
    print("test_basic_flow passed. boss hp:", game.boss.hp)


def test_clash_flow():
    game = GameUI()
    game.draw()

    unit = game.units[0]
    game._arm(unit.name, 1)  # arm the TOP slot this time
    game._commit_action(clash_slot_index=0)
    assert game.player_actions[unit.name].clash_slot_index == 0
    assert game.player_actions[unit.name].armed_position == 1
    assert 0 in game._claimed_slot_indices()

    for unit2 in game.units[1:]:
        game._arm(unit2.name, 0)
        game._commit_action(clash_slot_index=None)

    assert game._all_actions_ready()
    game._start_turn()
    game.draw()
    print("test_clash_flow passed.")


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
    game.draw()  # skill circles should be clickable again now
    clickable_positions = [
        c.payload[1][1] for c in game.clickables
        if c.payload[0] == "arm_skill" and c.payload[1][0] == unit.name
    ]
    assert set(clickable_positions) == {0, 1}
    print("test_cancel_action passed.")


def test_duplicate_skill_type_independent_arming():
    """
    Force a unit's available pair to be the SAME skill type, then confirm
    arming one position doesn't visually/logically arm the other, and that
    queue.pick correctly uses position to disambiguate the shift behavior.
    """
    game = GameUI()
    unit = game.units[0]
    # force available = ['skill1', 'skill1']
    unit.queue.queue[0] = 'skill1'
    unit.queue.queue[1] = 'skill1'
    assert unit.queue.available == ['skill1', 'skill1']

    game.draw()
    game._arm(unit.name, 1)  # arm TOP specifically
    assert game.armed == (unit.name, 1)
    is_armed_0 = game.armed == (unit.name, 0)
    is_armed_1 = game.armed == (unit.name, 1)
    assert is_armed_0 is False and is_armed_1 is True

    queue_len_before = len(unit.queue.queue)
    game._commit_action(clash_slot_index=None)
    action = game.player_actions[unit.name]
    assert action.armed_position == 1

    unit.queue.pick(action.skill_type, position=action.armed_position)
    assert len(unit.queue.queue) == queue_len_before - 2, "picking TOP with duplicate types should shift by 2"
    print("test_duplicate_skill_type_independent_arming passed.")


def test_unopposed_button_click_path():
    game = GameUI()
    game.draw()
    unit = game.units[0]
    game._arm(unit.name, 0)
    game.draw()  # need can_click True to register the button as clickable
    btn = next(
        c for c in game.clickables
        if c.payload[0] == "attack_unopposed" and c.payload[1] == unit.name
    )
    game.handle_click(btn.rect.center)
    assert unit.name in game.player_actions
    assert game.player_actions[unit.name].clash_slot_index is None
    print("test_unopposed_button_click_path passed.")


def test_manual_boss_toggle_and_cycle():
    game = GameUI()
    game.manual_boss_mode = True
    game.draw()

    slot0_before = game.boss_slots[0].skill_def.name
    game._cycle_boss_skill(0)
    slot0_after = game.boss_slots[0].skill_def.name
    assert slot0_after != slot0_before

    if not game.boss_slots[0].skill_def.hits_all:
        target_before = game.boss_slots[0].target_names[0]
        game._cycle_boss_target(0)
        target_after = game.boss_slots[0].target_names[0]
        assert target_after != target_before or len(game._alive_unit_names()) == 1
    print("test_manual_boss_toggle_and_cycle passed.")


def test_dead_unit_excluded_from_clickables():
    game = GameUI()
    game.units[1].hp = 0
    game.draw()
    kinds_for_dead_unit = [
        c.payload for c in game.clickables
        if c.payload[0] == "arm_skill" and c.payload[1][0] == game.units[1].name
    ]
    assert kinds_for_dead_unit == []
    print("test_dead_unit_excluded_from_clickables passed.")


def test_connection_line_data_present_after_commit():
    game = GameUI()
    unit = game.units[0]
    game._arm(unit.name, 0)
    game._commit_action(clash_slot_index=1)
    game.draw()  # populates skill_circle_pos and boss_slot_rects
    assert (unit.name, 0) in game.skill_circle_pos
    assert 1 in game.boss_slot_rects
    print("test_connection_line_data_present_after_commit passed.")


if __name__ == "__main__":
    test_basic_flow()
    test_clash_flow()
    test_cancel_action()
    test_duplicate_skill_type_independent_arming()
    test_unopposed_button_click_path()
    test_manual_boss_toggle_and_cycle()
    test_dead_unit_excluded_from_clickables()
    test_connection_line_data_present_after_commit()
    print("\nALL TESTS PASSED")
