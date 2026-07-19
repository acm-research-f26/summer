import pytest

from src.instrument.storage import (
    append_response,
    assign_conditions,
    load_responses,
    new_response_id,
)


def make_row(**overrides):
    row = {
        "response_id": new_response_id(),
        "ts_iso": "2026-07-18T00:00:00",
        "participant_id": "p_test",
        "source": "synthetic",
        "screen_id": "S1",
        "condition": "manipulated",
        "image_shown": "x.png",
        "decision": 1,
        "rt_ms": 1200,
    }
    row.update(overrides)
    return row


def test_append_and_load_roundtrip(tmp_path):
    path = tmp_path / "responses.csv"
    append_response(make_row(), path=path)
    append_response(make_row(decision=0, condition="neutralized"), path=path)
    df = load_responses(path)
    assert len(df) == 2
    assert list(df["decision"]) == [1, 0]
    # header written exactly once
    assert path.read_text().count("response_id") == 1


def test_invalid_rows_rejected(tmp_path):
    path = tmp_path / "responses.csv"
    with pytest.raises(ValueError):
        append_response(make_row(condition="control"), path=path)
    with pytest.raises(ValueError):
        append_response(make_row(source="pilot"), path=path)
    with pytest.raises(ValueError):
        append_response(make_row(decision=2), path=path)
    bad = make_row()
    del bad["rt_ms"]
    with pytest.raises(ValueError):
        append_response(bad, path=path)
    assert not path.exists()  # nothing invalid ever landed


def test_assign_conditions_deterministic_and_valid():
    ids = [f"S{i}" for i in range(10)]
    a = assign_conditions("p1", ids, seed=42)
    b = assign_conditions("p1", ids, seed=42)
    assert a == b
    assert set(a) == set(ids)
    assert set(a.values()) <= {"manipulated", "neutralized"}
    # different participants get (almost surely) different assignments
    others = [assign_conditions(f"p{k}", ids, seed=42) for k in range(2, 12)]
    assert any(o != a for o in others)
