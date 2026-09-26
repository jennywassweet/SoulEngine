"""Shared fixtures for the test suite"""

import base64
import zlib

import pytest

from engine.llm.base import LLMResponse
from engine.state_manager import StateManager
from engine.chat_logger import ChatLogger


class FakeLLMProvider:
    """Stub LLMProvider (duck-types the Protocol) returning a canned response"""

    def __init__(self, content: str = ""):
        self.content = content
        self.calls: list[dict] = []

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        reasoning_effort: str | None = None,
        provider_routing: dict | None = None,
    ) -> LLMResponse:
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "provider_routing": provider_routing,
        })
        return LLMResponse(content=self.content)


@pytest.fixture
def fake_llm():
    return FakeLLMProvider()


class FakeEmbeddingProvider:
    """Stub EmbeddingProvider (duck-types the Protocol) - deterministic, no network.

    Bag-of-words: each word hashes (via crc32, not the salted builtin hash())
    into one of `dimensions` buckets, so texts sharing vocabulary end up with
    similar vectors - close enough to a real embedding for testing ranking
    and retrieval logic without calling an actual API.
    """

    def __init__(self, dimensions: int = 32):
        self.model_name = "fake-embed"
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in text.lower().split():
            bucket = zlib.crc32(word.encode("utf-8")) % self.dimensions
            vector[bucket] += 1.0
        return vector


@pytest.fixture
def fake_embedding_provider():
    return FakeEmbeddingProvider()


class FakeImageProvider:
    """Stub ImageProvider (duck-types the Protocol) - deterministic, no network.

    Returns fixed bytes for a valid 1x1 PNG and remembers the last prompt,
    so callers (PhotoDirector tests) can assert on what was passed without
    parsing image bytes.
    """

    # A real, minimal 1x1 transparent PNG - valid enough for anything that
    # inspects the bytes rather than just forwarding them.
    _PNG_1X1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    def __init__(self, image_bytes: bytes | None = None):
        self.model_name = "fake-image"
        self.image_bytes = image_bytes or self._PNG_1X1
        self.calls: list[dict] = []

    async def generate(self, prompt: str, reference_images: list[bytes] | None = None) -> bytes:
        self.calls.append({"prompt": prompt, "reference_images": reference_images})
        return self.image_bytes


@pytest.fixture
def fake_image_provider():
    return FakeImageProvider()


@pytest.fixture
def state_manager(tmp_path):
    return StateManager(tmp_path / "state.md")


@pytest.fixture
def chat_logger(tmp_path, state_manager):
    return ChatLogger(tmp_path / "chat_log.jsonl", state_manager=state_manager)
