"""
PhotoStore - flat-file storage for the photo engine

Pure storage: no knowledge of prompts, LLMs, or image providers. Records
every request PhotoDirector makes, including refusals (no image produced),
because photos.jsonl - not the images themselves - is the feature's main
artifact: it's how a character's reading of a scene is inspected over time,
the same role summaries.jsonl plays for compression.

Fully isolated from the text circuit's stores (ChatLogger, MemoryStore):
separate directory, separate id space, never read by anything else.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PhotoStore:
    """Stores photo requests (images + a JSONL record of every request, including refusals)"""

    def __init__(self, photos_dir: Path | str, jsonl_path: Path | str):
        """
        Initialize photo store, loading any existing records from disk

        Args:
            photos_dir: Directory image files are written to (<id>.png)
            jsonl_path: Path to the request-log file (one JSON object per line)
        """
        self.photos_dir = Path(photos_dir)
        self.jsonl_path = Path(jsonl_path)

        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self._entries: list[dict[str, Any]] = []
        if self.jsonl_path.exists():
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                self._entries = [json.loads(line) for line in f if line.strip()]

        # Independent of ChatLogger's message_id counter: the two circuits
        # share no state at all. The photo engine never reads or writes
        # chat_log.jsonl, memory, continuity or the compression counters -
        # an invariant, not an optimisation.
        self._next_id = max((e["id"] for e in self._entries), default=0) + 1

    def record(
        self,
        *,
        mode: str,
        hint: str,
        available: bool,
        prompt: str,
        note: str,
        model: str,
        image_model: str | None = None,
        cost: float | None = None,
        reference: str | None = None,
        image_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """
        Record one photo request and, if an image was produced, save it.

        Args:
            mode: Camera position - "shot" / "scene" / "selfie"
            hint: Viewer's prompt argument (may be "")
            available: Whether the scene produced a frame at all
            prompt: The image-generation prompt PhotoDirector wrote (may be
                "" on refusal)
            note: World remark - either flavor text alongside a delivered
                frame, or the in-world reason for a refusal
            model: Text describer model that produced prompt/note
            image_model: Image-generation model used, if any image call
                was made
            cost: Cost reported by the image provider, if any image call
                was made
            reference: File name of the likeness reference sent with the
                image call (a character's face artifact), or None if the
                frame was generated from text alone. Recorded because
                otherwise a frame made with a reference is
                indistinguishable from one made without it after the
                fact - the same reason refusals are recorded at all.
            image_bytes: Raw image bytes, if a frame was produced.
                image_bytes is None for a refusal (available=False) or for
                a call that errored before an image came back.

        Returns:
            The stored entry, including its assigned id and file name.
        """
        photo_id = self._next_id
        self._next_id += 1

        file_name = None
        if image_bytes is not None:
            file_name = f"{photo_id}.png"
            (self.photos_dir / file_name).write_bytes(image_bytes)

        entry = {
            "id": photo_id,
            "ts": datetime.now().isoformat(),
            "mode": mode,
            "hint": hint,
            "available": available,
            "prompt": prompt,
            "note": note,
            "model": model,
            "image_model": image_model,
            "cost": cost,
            "reference": reference,
            "file": file_name,
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._entries.append(entry)

        return entry

    def last(self) -> dict[str, Any] | None:
        """Most recent record, or None if the store is empty. Used by /again."""
        return self._entries[-1] if self._entries else None

    def read_last_n(self, n: int) -> list[dict[str, Any]]:
        """Most recent n records, oldest first. Used by /photos."""
        return self._entries[-n:] if self._entries else []

    def last_available(self) -> dict[str, Any] | None:
        """Most recent record with a delivered frame (available=True), or
        None if there isn't one. Used by /again - a refusal has no prompt
        to revise, so it isn't what "the last frame" means there."""
        for entry in reversed(self._entries):
            if entry["available"]:
                return entry
        return None
