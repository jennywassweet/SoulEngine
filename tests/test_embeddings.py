"""Tests for the EmbeddingProvider contract, via the Fake implementation.

No network calls here - engine.embeddings.openrouter.OpenRouterEmbeddingProvider
is verified manually against the real API instead.
"""

import math


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def test_embed_returns_one_vector_per_text(fake_embedding_provider):
    vectors = await fake_embedding_provider.embed(["hello world", "another text", "third"])
    assert len(vectors) == 3
    assert all(len(v) == fake_embedding_provider.dimensions for v in vectors)


async def test_embed_records_the_batch_call(fake_embedding_provider):
    await fake_embedding_provider.embed(["one", "two"])
    assert fake_embedding_provider.calls == [["one", "two"]]


async def test_embed_is_deterministic(fake_embedding_provider):
    first = await fake_embedding_provider.embed(["repeat this phrase"])
    second = await fake_embedding_provider.embed(["repeat this phrase"])
    assert first == second


async def test_similar_texts_are_closer_than_unrelated_ones(fake_embedding_provider):
    anchor, similar, unrelated = await fake_embedding_provider.embed(
        [
            "kitten chasing a ball of yarn",
            "a kitten chasing yarn in the sun",
            "quarterly tax filing deadline",
        ]
    )
    assert _cosine(anchor, similar) > _cosine(anchor, unrelated)


async def test_empty_text_yields_zero_vector(fake_embedding_provider):
    vectors = await fake_embedding_provider.embed([""])
    assert vectors[0] == [0.0] * fake_embedding_provider.dimensions
