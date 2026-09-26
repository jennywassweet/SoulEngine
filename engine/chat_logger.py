"""
Chat logger for append-only JSONL logging
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Literal

from engine.state_manager import StateManager

# Distinguishes "caller didn't pass message_id" (auto-assign a fresh one)
# from "caller explicitly passed message_id=None" (leave it unset, e.g. a
# legacy log entry being rewritten that never had an id) — plain `None` as
# the default can't tell these apart.
_UNSET = object()


class ChatLogger:
    """Logs chat messages to JSONL file"""

    def __init__(self, log_file: Path | str, state_manager: StateManager | None = None):
        """
        Initialize chat logger

        Args:
            log_file: Path to JSONL log file
            state_manager: If given, backs auto-assigned message_id with a
                counter persisted in state.md (survives restarts and
                compression deleting old messages from this file). Without
                it, log_message() only stamps a message_id when the caller
                supplies one explicitly.
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_manager = state_manager

        # Create file if doesn't exist
        if not self.log_file.exists():
            self.log_file.touch()

    def log_message(
        self,
        role: Literal["user", "assistant", "system"],
        content: str,
        user_id: str | None = None,
        char_count: int | None = None,
        message_id: int | None = _UNSET,
        timestamp: str | None = None,
    ) -> int | None:
        """
        Log a message to JSONL

        Args:
            role: Message role (user/assistant/system)
            content: Message content
            user_id: User ID for user messages
            char_count: Character count (calculated if not provided)
            message_id: Stable id for retrieval memory to key on later.
                Omit to auto-assign the next id (requires state_manager);
                pass an explicit value (including None) to preserve it
                as-is, e.g. when rewriting existing entries.
            timestamp: ISO timestamp to record. Defaults to now; rewrite()
                passes each entry's original value so compression and
                rollback don't restamp surviving messages with the moment
                the file happened to be rewritten.

        Returns:
            The message_id this entry was stamped with (None if none could
            be resolved) — so callers that also archive into retrieval
            memory can key that entry with the exact same id.
        """
        if message_id is _UNSET:
            message_id = (
                self.state_manager.increment_counter("next_message_id")
                if self.state_manager is not None
                else None
            )

        entry = self._build_entry(role, content, user_id, char_count, message_id, timestamp)

        # Append to JSONL
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return message_id

    @staticmethod
    def _build_entry(
        role: str,
        content: str,
        user_id: str | None,
        char_count: int | None,
        message_id: int | None,
        timestamp: str | None,
    ) -> dict:
        if char_count is None:
            char_count = len(content)

        entry = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "role": role,
            "content": content,
            "char_count": char_count,
        }

        if user_id:
            entry["user_id"] = user_id

        if message_id is not None:
            entry["message_id"] = message_id

        return entry

    def read_all(self) -> list[dict]:
        """
        Read all log entries

        Returns:
            List of log entries as dictionaries
        """
        if not self.log_file.exists():
            return []

        entries = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        return entries

    def read_last_n(self, n: int) -> list[dict]:
        """
        Read last N log entries

        Args:
            n: Number of entries to read

        Returns:
            List of last N log entries
        """
        all_entries = self.read_all()
        return all_entries[-n:] if all_entries else []

    def read_last_n_chars(self, max_chars: int) -> list[dict]:
        """
        Read log entries up to max_chars total

        Args:
            max_chars: Maximum total characters

        Returns:
            List of most recent entries that fit in budget
        """
        entries = self.read_all()
        result = []
        total_chars = 0

        # Read backwards until budget exhausted
        for entry in reversed(entries):
            char_count = entry.get("char_count", len(entry.get("content", "")))
            if total_chars + char_count > max_chars:
                break
            result.insert(0, entry)
            total_chars += char_count

        return result

    def get_total_chars(self) -> int:
        """
        Get total character count of all logged messages

        Returns:
            Total character count
        """
        entries = self.read_all()
        return sum(e.get("char_count", 0) for e in entries)

    def clear(self) -> None:
        """Clear the log file"""
        self.log_file.write_text("", encoding="utf-8")

    def rewrite(self, messages: list[dict]) -> None:
        """
        Replace the log's entire contents with `messages`, preserving each
        entry's own message_id/user_id/char_count rather than reassigning
        new ones. Shared by compression (dropping the oldest chunk) and the
        /back and /regen control commands (dropping/temporarily hiding the
        newest entries).

        Args:
            messages: Entries to keep, in order. Pass [] to empty the log.

        Writes atomically (temp file + rename) so a process killed
        mid-rewrite leaves the previous contents intact rather than a
        truncated or empty file — this method used to clear() the file
        and then append entries one at a time, which could lose the
        entire log if interrupted in between.
        """
        lines = [
            json.dumps(
                self._build_entry(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    user_id=msg.get("user_id"),
                    char_count=msg.get("char_count"),
                    message_id=msg.get("message_id"),
                    timestamp=msg.get("timestamp"),
                ),
                ensure_ascii=False,
            )
            for msg in messages
        ]
        content = "".join(line + "\n" for line in lines)

        tmp_path = self.log_file.with_suffix(self.log_file.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, self.log_file)
