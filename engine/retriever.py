"""
Retriever - builds a retrieval query from recent chat history, ranks
MemoryStore results by relevance x recency, and formats them into a
prompt block.
"""

import logging
from datetime import datetime
from typing import Any

from engine.embeddings.base import EmbeddingProvider
from engine.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class Retriever:
    """Turns recent chat history into a ranked, formatted block of recalled fragments.

    Failure handling is the caller's job, not this class's: retrieval is an
    enhancement that must never block a turn, but that "catch and warn"
    belongs at the
    Orchestrator call site (same place compression failures are swallowed
    today) so this class can stay a plain, easily-testable pipeline that
    raises naturally on embedding/search errors.
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider,
        config: dict[str, Any],
        role_names: dict[str, str] | None = None,
    ):
        """
        Args:
            store: MemoryStore to search
            embedding_provider: used to embed the query text
            config: the `memory:` section of soul/config.yaml
            role_names: role -> display name, same mapping PromptBuilder uses
        """
        self.store = store
        self.embedding_provider = embedding_provider
        self.role_names = role_names or {"user": "user", "assistant": "assistant"}

        self.top_k = config.get("top_k", 5)
        self.query_window_messages = config.get("query_window_messages", 4)
        self.recency_half_life_days = config.get("recency_half_life_days", 30)
        self.max_chars = config.get("max_chars", 1500)
        self.min_score = config.get("min_score", 0.0)

    def _format_line(self, role: str, content: str) -> str:
        name = self.role_names.get(role, role)
        return f"{name}: {content}"

    def build_query_text(self, recent_messages: list[dict], current_message: str) -> str:
        """Concatenate the last `query_window_messages` history entries with the
        current incoming message, formatted like the main manuscript."""
        window = recent_messages[-self.query_window_messages:] if self.query_window_messages > 0 else []
        lines = [self._format_line(m.get("role", "user"), m.get("content", "")) for m in window]
        lines.append(self._format_line("user", current_message))
        return "\n".join(lines)

    def _recency_weight(self, timestamp: str, now: datetime) -> float:
        if self.recency_half_life_days <= 0:
            return 1.0
        try:
            age_days = max((now - datetime.fromisoformat(timestamp)).total_seconds() / 86400, 0.0)
        except (TypeError, ValueError):
            return 1.0
        return 0.5 ** (age_days / self.recency_half_life_days)

    def rank(self, results: list[tuple[dict, float]], now: datetime) -> list[tuple[dict, float]]:
        """
        Re-score (entry, cosine) pairs as cosine * recency, sorted highest first.

        min_score gates on the raw cosine, before recency is applied: the
        question "is this fragment about the same thing at all" is separate
        from "how old is it". Gating the combined score instead would drop a
        genuinely relevant but old message (cosine 0.5 at two half-lives =
        0.125) as if it were off-topic, and make the knob untunable.
        """
        scored = [
            (entry, cosine * self._recency_weight(entry.get("timestamp", ""), now))
            for entry, cosine in results
            if cosine >= self.min_score
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def _trim_to_budget(self, entries: list[dict]) -> list[dict]:
        """Keep whole entries within max_chars, in the given (rank) order.

        Measures each entry as the line format_block() will actually emit
        ("Name: text"), not the bare content — otherwise max_chars
        undercounts by the name prefix on every line.
        """
        kept = []
        total = 0
        for entry in entries:
            line = self._format_line(entry.get("role", "user"), entry.get("content", ""))
            if total + len(line) > self.max_chars:
                break
            kept.append(entry)
            total += len(line)
        return kept

    async def retrieve(
        self,
        recent_messages: list[dict],
        current_message: str,
        exclude_ids: set[int] | None = None,
        now: datetime | None = None,
    ) -> list[dict]:
        """
        Full pipeline: build query text -> embed -> search the whole store ->
        rank by relevance x recency -> cut to top_k -> trim to max_chars ->
        return in chronological order (oldest first), so the block reads as
        excerpts, not a leaderboard.
        """
        query_text = self.build_query_text(recent_messages, current_message)
        [query_vector] = await self.embedding_provider.embed([query_text])
        return await self.retrieve_with_vector(query_vector, exclude_ids=exclude_ids, now=now)

    async def retrieve_with_vector(
        self,
        query_vector: list[float],
        exclude_ids: set[int] | None = None,
        now: datetime | None = None,
    ) -> list[dict]:
        """
        Same pipeline as retrieve(), but takes an already-computed query
        vector. Lets a caller batch the query embedding together with some
        other text (e.g. the incoming message being archived) in a single
        embedding-API call instead of retrieve() making its own.

        Each returned entry carries an extra "score" key (its final
        relevance x recency score) - a shallow copy, so this never mutates
        the entries actually held by the MemoryStore.
        """
        now = now or datetime.now()

        # Rank over the whole store, not just an embedding-search top_k: brute
        # force cosine is cheap at our scale, and a recent-but-slightly-less-
        # similar message can only outrank an old-but-closer one if it was in
        # the candidate pool to begin with.
        raw_results = self.store.search(query_vector, top_k=len(self.store), exclude_ids=exclude_ids)
        ranked = self.rank(raw_results, now)
        selected = [{**entry, "score": score} for entry, score in ranked[: self.top_k]]

        trimmed = self._trim_to_budget(selected)
        trimmed.sort(key=lambda e: e.get("message_id", 0))
        return trimmed

    def format_block(self, entries: list[dict], header: str = "") -> str:
        """Format entries the same way the manuscript is formatted: 'Name: text'."""
        if not entries:
            return ""
        body = "\n".join(self._format_line(e.get("role", "user"), e.get("content", "")) for e in entries)
        return f"{header}\n\n{body}" if header else body
