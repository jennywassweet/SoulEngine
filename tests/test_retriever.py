"""Tests for engine.retriever.Retriever"""

from datetime import datetime, timedelta

import pytest

from engine.memory_store import MemoryStore
from engine.retriever import Retriever

NOW = datetime(2026, 9, 6, 12, 0, 0)
ROLE_NAMES = {"user": "Misha", "assistant": "Zoya"}


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        jsonl_path=tmp_path / "memory.jsonl",
        npy_path=tmp_path / "memory_vectors.npy",
        model_name="fake-embed",
    )


@pytest.fixture
def retriever(store, fake_embedding_provider):
    config = {"top_k": 5, "query_window_messages": 4, "recency_half_life_days": 30, "max_chars": 1500}
    return Retriever(store, fake_embedding_provider, config, role_names=ROLE_NAMES)


async def _seed(store, fake_embedding_provider, entries: list[dict]) -> None:
    """entries: list of {message_id, content, role, timestamp}"""
    vectors = await fake_embedding_provider.embed([e["content"] for e in entries])
    store.add(
        entries=[
            {
                "message_id": e["message_id"],
                "timestamp": e["timestamp"],
                "role": e.get("role", "user"),
                "content": e["content"],
            }
            for e in entries
        ],
        vectors=vectors,
    )


def _iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


# --- build_query_text ---

def test_build_query_text_respects_window_size(retriever):
    history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    query = retriever.build_query_text(history, "current")

    lines = query.split("\n")
    # window (4) + current message
    assert len(lines) == 5
    assert lines[-1] == "Misha: current"
    assert lines[0] == "Misha: msg 6"


def test_build_query_text_zero_window_uses_only_current(retriever):
    retriever.query_window_messages = 0
    history = [{"role": "user", "content": "old"}]
    query = retriever.build_query_text(history, "current")
    assert query == "Misha: current"


# --- retrieve() ---

async def test_retrieve_on_empty_store_returns_nothing(retriever):
    result = await retriever.retrieve(recent_messages=[], current_message="the birthday secret")
    assert result == []


async def test_freshest_wins_at_equal_similarity(store, fake_embedding_provider, retriever):
    await _seed(
        store,
        fake_embedding_provider,
        [
            {"message_id": 1, "content": "the birthday secret about zoya", "timestamp": _iso(days_ago=60)},
            {"message_id": 2, "content": "the birthday secret about zoya", "timestamp": _iso(days_ago=1)},
        ],
    )
    retriever.top_k = 1

    result = await retriever.retrieve(
        recent_messages=[], current_message="the birthday secret about zoya", now=NOW
    )

    assert [e["message_id"] for e in result] == [2]


async def test_exclude_ids_are_never_returned(store, fake_embedding_provider, retriever):
    await _seed(
        store,
        fake_embedding_provider,
        [
            {"message_id": 1, "content": "favourite topic about coffee", "timestamp": _iso(days_ago=1)},
            {"message_id": 2, "content": "favourite topic about coffee", "timestamp": _iso(days_ago=1)},
        ],
    )

    result = await retriever.retrieve(
        recent_messages=[],
        current_message="favourite topic about coffee",
        exclude_ids={1, 2},
        now=NOW,
    )

    assert result == []


async def test_max_chars_keeps_whole_messages_only(store, fake_embedding_provider, retriever):
    await _seed(
        store,
        fake_embedding_provider,
        [
            {"message_id": 1, "content": "a" * 40, "timestamp": _iso(days_ago=1)},
            {"message_id": 2, "content": "a" * 40, "timestamp": _iso(days_ago=1)},
        ],
    )
    retriever.max_chars = 50  # fits exactly one 40-char message, not two

    result = await retriever.retrieve(recent_messages=[], current_message="a" * 40, now=NOW)

    assert len(result) == 1
    assert result[0]["content"] == "a" * 40


async def test_retrieve_returns_chronological_order(store, fake_embedding_provider, retriever):
    # id=3 is the closest match to the query but the oldest of the three;
    # id=1 is the weakest match but newest. All three should still come
    # back sorted by message_id, not by relevance rank.
    await _seed(
        store,
        fake_embedding_provider,
        [
            {"message_id": 1, "content": "the weather today is sunny", "timestamp": _iso(days_ago=0.1)},
            {"message_id": 2, "content": "favourite film about space", "timestamp": _iso(days_ago=30)},
            {"message_id": 3, "content": "favourite film about space and stars", "timestamp": _iso(days_ago=60)},
        ],
    )
    retriever.top_k = 3
    retriever.recency_half_life_days = 0  # pure relevance, no recency reordering

    result = await retriever.retrieve(
        recent_messages=[], current_message="favourite film about space and stars", now=NOW
    )

    assert [e["message_id"] for e in result] == [1, 2, 3]


async def test_min_score_drops_weak_matches(store, fake_embedding_provider, retriever):
    await _seed(
        store,
        fake_embedding_provider,
        [
            {"message_id": 1, "content": "favourite film about space", "timestamp": _iso(days_ago=1)},
            {"message_id": 2, "content": "a completely different topic", "timestamp": _iso(days_ago=1)},
        ],
    )
    # the query carries a "Misha: " prefix the stored content doesn't, so
    # even an exact text match lands near 0.89 with the fake provider
    retriever.min_score = 0.85  # only a near-exact match survives

    result = await retriever.retrieve(
        recent_messages=[], current_message="favourite film about space", now=NOW
    )

    assert [e["message_id"] for e in result] == [1]


async def test_min_score_gates_on_cosine_not_on_recency(store, fake_embedding_provider, retriever):
    """An old but highly relevant message must survive the gate — recency
    scales the ranking, it doesn't make something off-topic."""
    await _seed(
        store,
        fake_embedding_provider,
        [{"message_id": 1, "content": "favourite film about space", "timestamp": _iso(days_ago=300)}],
    )
    # the query carries a "Misha: " prefix the stored content doesn't, so
    # even an exact text match lands near 0.89 with the fake provider
    retriever.min_score = 0.85

    result = await retriever.retrieve(
        recent_messages=[], current_message="favourite film about space", now=NOW
    )

    assert [e["message_id"] for e in result] == [1]
    assert result[0]["score"] < 0.01  # heavily discounted by age, but still returned


async def test_trim_to_budget_counts_the_name_prefix(store, fake_embedding_provider, retriever):
    await _seed(
        store,
        fake_embedding_provider,
        [
            {"message_id": 1, "content": "a" * 20, "timestamp": _iso(days_ago=1)},
            {"message_id": 2, "content": "a" * 20, "timestamp": _iso(days_ago=1)},
        ],
    )
    # Two bare contents (40) would fit in 45, but with "Misha: " prefixes
    # each line is 27 chars, so only one fits.
    retriever.max_chars = 45

    result = await retriever.retrieve(recent_messages=[], current_message="a" * 20, now=NOW)

    assert len(result) == 1


# --- retrieve_with_vector() ---

async def test_retrieve_with_vector_matches_retrieve_given_the_same_query(
    store, fake_embedding_provider, retriever
):
    await _seed(
        store,
        fake_embedding_provider,
        [{"message_id": 1, "content": "favourite topic about coffee", "timestamp": _iso(days_ago=1)}],
    )

    via_retrieve = await retriever.retrieve(recent_messages=[], current_message="coffee", now=NOW)

    [vector] = await fake_embedding_provider.embed(
        [retriever.build_query_text([], "coffee")]
    )
    via_vector = await retriever.retrieve_with_vector(vector, now=NOW)

    assert [e["message_id"] for e in via_retrieve] == [e["message_id"] for e in via_vector]


async def test_retrieve_with_vector_exposes_score_without_mutating_the_store(
    store, fake_embedding_provider, retriever
):
    await _seed(
        store,
        fake_embedding_provider,
        [{"message_id": 1, "content": "favourite topic about coffee", "timestamp": _iso(days_ago=1)}],
    )
    [vector] = await fake_embedding_provider.embed(["coffee"])

    result = await retriever.retrieve_with_vector(vector, now=NOW)

    assert "score" in result[0]
    # the store's own copy must stay untouched by the caller-facing "score" key
    stored_entry = store.search(vector, top_k=1)[0][0]
    assert "score" not in stored_entry


# --- format_block ---

def test_format_block_empty_entries_returns_empty_string(retriever):
    assert retriever.format_block([]) == ""


def test_format_block_matches_manuscript_style(retriever):
    entries = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hi!"},
    ]
    assert retriever.format_block(entries) == "Misha: hi\nZoya: hi!"


def test_format_block_prefixes_header(retriever):
    result = retriever.format_block([{"role": "user", "content": "hi"}], header="Earlier:")
    assert result == "Earlier:\n\nMisha: hi"
