"""Tests for engine.photo_store.PhotoStore"""

import json

import pytest

from engine.photo_store import PhotoStore


@pytest.fixture
def store(tmp_path):
    return PhotoStore(photos_dir=tmp_path / "photos", jsonl_path=tmp_path / "photos.jsonl")


def _delivered(store, **overrides):
    kwargs = dict(
        mode="shot",
        hint="",
        available=True,
        prompt="a phone snapshot",
        note="",
        model="describer-model",
        image_model="image-model",
        cost=0.06,
        image_bytes=b"fake-png-bytes",
    )
    kwargs.update(overrides)
    return store.record(**kwargs)


def _refused(store, **overrides):
    kwargs = dict(
        mode="selfie",
        hint="right now",
        available=False,
        prompt="",
        note="she is in the dark and won't turn on the light",
        model="describer-model",
    )
    kwargs.update(overrides)
    return store.record(**kwargs)


def test_delivered_photo_writes_file_and_entry(store):
    entry = _delivered(store)

    assert entry["id"] == 1
    assert entry["available"] is True
    assert entry["file"] == "1.png"
    assert (store.photos_dir / "1.png").read_bytes() == b"fake-png-bytes"


def test_refusal_writes_no_file(store):
    entry = _refused(store)

    assert entry["available"] is False
    assert entry["file"] is None
    assert entry["image_model"] is None
    assert entry["cost"] is None
    assert list(store.photos_dir.iterdir()) == []


def test_refusal_is_still_recorded_in_jsonl(store):
    _refused(store)

    lines = store.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["available"] is False
    assert logged["note"] == "she is in the dark and won't turn on the light"


def test_id_counter_is_monotonic_across_refusals_and_deliveries(store):
    first = _delivered(store)
    second = _refused(store)
    third = _delivered(store)

    assert [first["id"], second["id"], third["id"]] == [1, 2, 3]


def test_last_returns_most_recent_record(store):
    _delivered(store)
    second = _refused(store)

    assert store.last() == second


def test_last_on_empty_store_returns_none(store):
    assert store.last() is None


def test_last_available_skips_a_trailing_refusal(store):
    delivered = _delivered(store)
    _refused(store)

    assert store.last_available() == delivered


def test_last_available_on_empty_or_all_refused_store_returns_none(store):
    assert store.last_available() is None
    _refused(store)
    assert store.last_available() is None


def test_read_last_n_returns_oldest_first(store):
    first = _delivered(store)
    second = _refused(store)
    third = _delivered(store)

    assert store.read_last_n(2) == [second, third]
    assert store.read_last_n(10) == [first, second, third]


def test_reload_from_disk_resumes_counter_and_content(tmp_path):
    store = PhotoStore(photos_dir=tmp_path / "photos", jsonl_path=tmp_path / "photos.jsonl")
    _delivered(store)
    _delivered(store)

    reloaded = PhotoStore(photos_dir=tmp_path / "photos", jsonl_path=tmp_path / "photos.jsonl")
    assert reloaded.last()["id"] == 2
    third = _delivered(reloaded)
    assert third["id"] == 3


def test_separate_settings_are_isolated(tmp_path):
    store_a = PhotoStore(photos_dir=tmp_path / "a" / "photos", jsonl_path=tmp_path / "a" / "photos.jsonl")
    store_b = PhotoStore(photos_dir=tmp_path / "b" / "photos", jsonl_path=tmp_path / "b" / "photos.jsonl")

    _delivered(store_a)
    _delivered(store_a)

    assert store_b.last() is None
    entry_b = _delivered(store_b)
    assert entry_b["id"] == 1
