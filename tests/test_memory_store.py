"""Tests for engine.memory_store.MemoryStore"""

import json

import pytest

from engine.memory_store import MemoryStore


def _entry(message_id: int, content: str = "text", role: str = "user") -> dict:
    return {"message_id": message_id, "timestamp": "2026-09-06T00:00:00", "role": role, "content": content}


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        jsonl_path=tmp_path / "memory.jsonl",
        npy_path=tmp_path / "memory_vectors.npy",
        model_name="fake-embed",
    )


def test_empty_store_len_and_search(store):
    assert len(store) == 0
    assert store.all_ids() == []
    assert store.search(query_vector=[1.0, 0.0, 0.0], top_k=5) == []


def test_add_then_search_returns_nearest_first(store):
    store.add(
        entries=[_entry(1, "a"), _entry(2, "b"), _entry(3, "c, close to a")],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
    )

    results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=3)

    assert [entry["message_id"] for entry, _score in results] == [1, 3, 2]
    assert results[0][1] > results[1][1] > results[2][1]


def test_search_respects_top_k(store):
    store.add(
        entries=[_entry(1), _entry(2), _entry(3)],
        vectors=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
    )
    results = store.search(query_vector=[1.0, 0.0], top_k=2)
    assert len(results) == 2


def test_search_respects_exclude_ids(store):
    store.add(
        entries=[_entry(1), _entry(2)],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
    )
    results = store.search(query_vector=[1.0, 0.0], top_k=5, exclude_ids={1})
    assert [entry["message_id"] for entry, _score in results] == [2]


def test_add_stamps_model_name(store):
    store.add(entries=[_entry(1)], vectors=[[1.0, 0.0]])
    assert store.all_ids() == [1]
    stored = store.search(query_vector=[1.0, 0.0], top_k=1)[0][0]
    assert stored["model"] == "fake-embed"


def test_add_mismatched_lengths_raises(store):
    with pytest.raises(ValueError):
        store.add(entries=[_entry(1), _entry(2)], vectors=[[1.0, 0.0]])


def test_add_empty_list_is_noop(store):
    store.add(entries=[], vectors=[])
    assert len(store) == 0


def test_delete_from_removes_matching_and_newer(store):
    store.add(
        entries=[_entry(1), _entry(2), _entry(3), _entry(4)],
        vectors=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    )
    store.delete_from(3)
    assert store.all_ids() == [1, 2]


def test_delete_from_can_empty_the_store(store):
    store.add(entries=[_entry(1)], vectors=[[1.0, 0.0]])
    store.delete_from(1)
    assert len(store) == 0
    assert store.search(query_vector=[1.0, 0.0], top_k=5) == []


def test_clear_empties_store(store):
    store.add(entries=[_entry(1), _entry(2)], vectors=[[1.0, 0.0], [0.0, 1.0]])
    store.clear()
    assert len(store) == 0
    assert store.all_ids() == []


def test_reload_from_disk_gives_same_content(tmp_path):
    jsonl_path = tmp_path / "memory.jsonl"
    npy_path = tmp_path / "memory_vectors.npy"

    first = MemoryStore(jsonl_path, npy_path, model_name="fake-embed")
    first.add(entries=[_entry(1, "hello"), _entry(2, "world")], vectors=[[1.0, 0.0], [0.0, 1.0]])

    reloaded = MemoryStore(jsonl_path, npy_path, model_name="fake-embed")

    assert reloaded.all_ids() == [1, 2]
    results = reloaded.search(query_vector=[1.0, 0.0], top_k=1)
    assert results[0][0]["content"] == "hello"


def test_reload_after_delete_from_persists_the_deletion(tmp_path):
    jsonl_path = tmp_path / "memory.jsonl"
    npy_path = tmp_path / "memory_vectors.npy"

    first = MemoryStore(jsonl_path, npy_path, model_name="fake-embed")
    first.add(entries=[_entry(1), _entry(2)], vectors=[[1.0, 0.0], [0.0, 1.0]])
    first.delete_from(2)

    reloaded = MemoryStore(jsonl_path, npy_path, model_name="fake-embed")
    assert reloaded.all_ids() == [1]


def test_detects_desynced_files_and_refuses_writes(tmp_path, caplog):
    """A crash between the jsonl append and the .npy write leaves the pair
    out of sync; that must not turn into an IndexError on every search."""
    jsonl_path = tmp_path / "memory.jsonl"
    npy_path = tmp_path / "memory_vectors.npy"

    store = MemoryStore(jsonl_path, npy_path, model_name="fake-embed")
    store.add(entries=[_entry(1), _entry(2)], vectors=[[1.0, 0.0], [0.0, 1.0]])

    # simulate the crash window: text row written, vector row not
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_entry(3)) + "\n")

    with caplog.at_level("ERROR"):
        reloaded = MemoryStore(jsonl_path, npy_path, model_name="fake-embed")

    assert reloaded.degraded is True
    assert "out of sync" in caplog.text
    # reads still work over the consistent prefix, no IndexError
    assert [e["message_id"] for e, _ in reloaded.search([1.0, 0.0], top_k=5)] == [1, 2]
    # and writes are refused so the gap can't grow
    reloaded.add(entries=[_entry(4)], vectors=[[1.0, 0.0]])
    assert len(reloaded) == 2
    # the text file is left intact — reindex needs it to rebuild the vectors
    assert len(jsonl_path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_warns_on_model_mismatch(tmp_path, caplog):
    jsonl_path = tmp_path / "memory.jsonl"
    npy_path = tmp_path / "memory_vectors.npy"

    built_with_old_model = MemoryStore(jsonl_path, npy_path, model_name="old-model")
    built_with_old_model.add(entries=[_entry(1)], vectors=[[1.0, 0.0]])

    with caplog.at_level("WARNING"):
        MemoryStore(jsonl_path, npy_path, model_name="new-model")

    assert "old-model" in caplog.text
    assert "reindex" in caplog.text
