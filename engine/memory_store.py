"""
MemoryStore - flat-file storage for retrieval memory (embedded messages)

Pure storage: no knowledge of LLMs, prompts, or embedding providers. Caller
supplies vectors already computed; this module only persists them alongside
their text/metadata and answers nearest-neighbor queries.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MemoryStore:
    """Stores embedded messages (text + vector) and searches them by cosine similarity"""

    def __init__(self, jsonl_path: Path | str, npy_path: Path | str, model_name: str):
        """
        Initialize memory store, loading any existing data from disk

        Args:
            jsonl_path: Path to the text+metadata file (one JSON object per line)
            npy_path: Path to the matching vector matrix (row i <-> jsonl line i)
            model_name: Embedding model this store's vectors are expected to be from
        """
        self.jsonl_path = Path(jsonl_path)
        self.npy_path = Path(npy_path)
        self.model_name = model_name

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.npy_path.parent.mkdir(parents=True, exist_ok=True)

        self._entries: list[dict[str, Any]] = []
        if self.jsonl_path.exists():
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                self._entries = [json.loads(line) for line in f if line.strip()]

        if self.npy_path.exists():
            self._vectors = np.load(self.npy_path)
        else:
            self._vectors = np.empty((0, 0), dtype=np.float32)

        # add() appends to the jsonl and rewrites the .npy as two separate
        # writes, so a crash between them leaves the pair out of sync — and
        # an out-of-sync store makes search() raise IndexError on every
        # single turn, which the caller only sees as a swallowed warning
        # (retrieval silently dead). Detect it here instead: read from the
        # prefix both files agree on, and refuse further writes so the gap
        # can't grow. Neither file is rewritten — memory.jsonl holds the
        # text the reindex script needs to rebuild the vectors from.
        self.degraded = len(self._entries) != len(self._vectors)
        if self.degraded:
            usable = min(len(self._entries), len(self._vectors))
            logger.error(
                f"MemoryStore out of sync at {self.jsonl_path}: {len(self._entries)} entries "
                f"vs {len(self._vectors)} vectors. Serving the first {usable} read-only and "
                f"refusing writes — run the reindex script to rebuild."
            )
            self._entries = self._entries[:usable]
            self._vectors = self._vectors[:usable]

        other_models = {e["model"] for e in self._entries if e.get("model") != model_name}
        if other_models:
            logger.warning(
                f"MemoryStore at {self.jsonl_path} built with model(s) {other_models}, "
                f"config says {model_name} — run reindex_memory.py"
            )

    def __len__(self) -> int:
        return len(self._entries)

    def all_ids(self) -> list[int]:
        return [e["message_id"] for e in self._entries]

    def add(self, entries: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        """
        Append entries and their vectors. Stamps each entry with this
        store's model_name (callers don't set it themselves).

        Args:
            entries: One dict per message (message_id, timestamp, role, content, ...)
            vectors: One embedding vector per entry, same order
        """
        if len(entries) != len(vectors):
            raise ValueError(f"entries/vectors length mismatch: {len(entries)} != {len(vectors)}")
        if not entries:
            return
        if self.degraded:
            logger.warning("MemoryStore is degraded (see startup error) — refusing to add")
            return

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            for entry in entries:
                entry = {**entry, "model": self.model_name}
                self._entries.append(entry)
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        new_vectors = _normalize_rows(np.array(vectors, dtype=np.float32))
        if self._vectors.size == 0:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])
        np.save(self.npy_path, self._vectors)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        exclude_ids: set[int] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Find the top_k entries most similar to query_vector by cosine similarity.

        Args:
            query_vector: Embedding to search against
            top_k: Maximum number of results
            exclude_ids: message_ids to skip (e.g. ones already in the manuscript)

        Returns:
            (entry, score) pairs, highest score first
        """
        if len(self._entries) == 0:
            return []

        exclude_ids = exclude_ids or set()
        q = _normalize_rows(np.array([query_vector], dtype=np.float32))[0]
        scores = self._vectors @ q

        candidates = [
            (self._entries[i], float(scores[i]))
            for i in range(len(self._entries))
            if self._entries[i]["message_id"] not in exclude_ids
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates[:top_k]

    def delete_from(self, message_id: int) -> None:
        """Delete all entries with message_id >= message_id (for rollback)."""
        keep_indices = [i for i, e in enumerate(self._entries) if e["message_id"] < message_id]
        dim = self._vectors.shape[1] if self._vectors.ndim == 2 else 0

        self._entries = [self._entries[i] for i in keep_indices]
        self._vectors = self._vectors[keep_indices] if keep_indices else np.empty((0, dim), dtype=np.float32)
        self._rewrite()

    def clear(self) -> None:
        """Delete everything (for /reset)."""
        self._entries = []
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._rewrite()

    def _rewrite(self) -> None:
        """Rewrite both files in full to match in-memory state."""
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        np.save(self.npy_path, self._vectors)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row; rows with zero norm are left as-is."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
