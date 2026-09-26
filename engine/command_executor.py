"""
Execution module for control commands parsed by command_parser.

/reset ports reset-setting.sh's "clean start" semantics so they're available
live, from inside the running bot, without shelling out. This is safe to do
mid-session because StateManager and ChatLogger hold no in-memory cache —
every read/write goes to disk, so truncating the files out from under a
running Orchestrator doesn't leave stale state anywhere.

/switch is not the same kind of "safe mid-session" operation: Orchestrator
itself caches a whole tree of setting-scoped objects (PromptBuilder,
Compressor, MemoryStore, the LLM provider...) in instance attributes at
construction time, so there is no live object to mutate in place. /switch
therefore writes the new setting into soul/config.yaml and sets
`pending_restart`; Orchestrator checks that flag after sending this
command's response and, if set, re-execs the whole process (see
`_restart_process` in orchestrator.py) so every object gets rebuilt from
scratch against the new setting — simpler and more reliable than trying to
hot-swap each cached object individually.
"""

import logging
from datetime import datetime
from pathlib import Path

from engine.chat_logger import ChatLogger
from engine.command_parser import ParsedCommand
from engine.config_utils import (
    list_settings,
    read_config,
    read_setting_characters,
    resolve_llm_config,
    write_active_setting,
)
from engine.memory_store import MemoryStore

logger = logging.getLogger(__name__)


def _split_trailing_pairs(messages: list[dict], count: int) -> tuple[list[dict], list[dict]]:
    """
    Pop `count` (user, assistant) exchanges from the tail of `messages`.

    A trailing message with no reply (the log doesn't end in "assistant" —
    a prior turn's LLM call failed after logging the incoming message) is
    not a completed exchange, so it's always dropped first and doesn't
    count against `count`.

    Returns:
        (remaining, removed) — removed is in original chronological order.
    """
    remaining = list(messages)
    removed: list[dict] = []

    if remaining and remaining[-1].get("role") != "assistant":
        removed.insert(0, remaining.pop())

    popped = 0
    while popped < count and remaining:
        removed.insert(0, remaining.pop())  # the assistant reply
        if remaining and remaining[-1].get("role") == "user":
            removed.insert(0, remaining.pop())  # the message it replied to
        popped += 1

    return remaining, removed


class CommandExecutor:
    """Runs control commands against a setting's runtime files."""

    def __init__(
        self,
        chat_logger: ChatLogger,
        state_path: Path | str,
        user_model_path: Path | str,
        continuity_path: Path | str,
        summaries_path: Path | str,
        opening_line_path: Path | str,
        debug_dir: Path | str,
        memory_store: MemoryStore | None = None,
        base_path: Path | str | None = None,
        setting: str | None = None,
    ):
        self.chat_logger = chat_logger
        self.state_path = Path(state_path)
        self.user_model_path = Path(user_model_path)
        self.continuity_path = Path(continuity_path)
        self.summaries_path = Path(summaries_path)
        self.opening_line_path = Path(opening_line_path)
        self.debug_dir = Path(debug_dir)
        self.memory_store = memory_store
        self.base_path = Path(base_path) if base_path is not None else None
        self.setting = setting
        # Set by _switch(); Orchestrator checks this after sending this
        # command's response and re-execs the process if it's True.
        self.pending_restart = False

    async def execute(self, command: ParsedCommand) -> str:
        if command.name == "reset":
            return self._reset()
        if command.name == "back":
            return self._back(command.args)
        if command.name == "list":
            return self._list()
        if command.name == "switch":
            return self._switch(command.args)
        return f"Unknown control command: /{command.name}"

    def _list(self) -> str:
        """List available settings, marking the currently active one."""
        settings = list_settings(self.base_path)
        if not settings:
            return "No settings found under soul/settings/."
        lines = ["📋 **Available settings**\n"]
        for name in settings:
            marker = " ← current" if name == self.setting else ""
            lines.append(f"  - {name}{marker}")
        return "\n".join(lines)

    def _switch(self, args: list[str]) -> str:
        """Switch the active setting, writing it and the setting's own
        self/other pairing into soul/config.yaml, then flag Orchestrator to
        restart the process so the switch actually takes effect.

        Takes the setting name and nothing else: the pairing is read from the
        setting's characters.yaml rather than passed in, so it can't be
        carried over from the previously active setting (stale slugs load as
        empty bibles, silently) or flipped mid-history (chat_log.jsonl stores
        role, not name — flipping self/other relabels every past message)."""
        if not args:
            return "Usage: /switch <setting>\n\n" + self._list()

        setting = args[0]
        setting_path = self.base_path / "soul" / "settings" / setting
        if not setting_path.is_dir():
            return f"❌ Unknown setting: {setting}\n\n" + self._list()

        characters = read_setting_characters(self.base_path, setting)
        missing = [key for key in ("self", "other") if key not in characters]
        if missing:
            return (
                f"❌ Setting '{setting}' doesn't declare {' and '.join(missing)} "
                f"in soul/settings/{setting}/characters.yaml — can't switch without it."
            )

        for slug in (characters["self"], characters["other"]):
            if not (setting_path / "characters" / slug).is_dir():
                return (
                    f"❌ Setting '{setting}' declares character '{slug}', but "
                    f"soul/settings/{setting}/characters/{slug}/ doesn't exist."
                )

        base_llm_config = read_config(self.base_path).get("llm", {})
        resolved_llm = resolve_llm_config(self.base_path, setting, base_llm_config)
        write_active_setting(self.base_path, setting, characters["self"], characters["other"])
        self.pending_restart = True

        logger.info(f"Switching setting: {self.setting} -> {setting}")
        return (
            f"🔄 Switching to **{setting}**\n"
            f"Writing as: {characters['self']} (to {characters['other']})\n"
            f"Model: {resolved_llm.get('provider', 'mistral')}/{resolved_llm.get('model', 'default')}\n"
            f"Temperature: {resolved_llm.get('temperature', 0.7)}\n\n"
            "Restarting now — send your next message once the bot reconnects."
        )

    def _back(self, args: list[str]) -> str:
        """Roll back the last N (user, assistant) exchanges."""
        if len(args) != 1 or not args[0].isdigit() or int(args[0]) <= 0:
            return "Usage: /back N (N = number of exchanges to roll back, N > 0)"

        count = int(args[0])
        all_messages = self.chat_logger.read_all()
        remaining, removed = _split_trailing_pairs(all_messages, count)

        if not removed:
            return "Nothing to roll back — chat log is empty."

        self.chat_logger.rewrite(remaining)

        removed_ids = [m["message_id"] for m in removed if m.get("message_id") is not None]
        if removed_ids and self.memory_store is not None:
            self.memory_store.delete_from(min(removed_ids))

        state_manager = self.chat_logger.state_manager
        if state_manager is not None:
            state_manager.update_frontmatter({
                "total_chars_since_last_compression": self.chat_logger.get_total_chars(),
                "last_updated": datetime.now().isoformat(),
            })

        pairs_removed = sum(1 for m in removed if m.get("role") == "assistant")
        logger.info(f"Rolled back {pairs_removed} exchange(s), {len(removed)} message(s) removed")
        return (
            f"✅ Rolled back {pairs_removed} exchange(s) ({len(removed)} message(s) removed). "
            f"{len(remaining)} message(s) remain. Note: continuity.md is not rolled back."
        )

    def _reset(self) -> str:
        """Wipe a setting's runtime state back to a clean start."""
        for path in (self.state_path, self.user_model_path, self.continuity_path):
            path.unlink(missing_ok=True)

        self.summaries_path.parent.mkdir(parents=True, exist_ok=True)
        self.summaries_path.write_text("", encoding="utf-8")

        for name in ("last_main_prompt.txt", "last_compression_prompt.txt", "prompt_log.jsonl"):
            (self.debug_dir / name).unlink(missing_ok=True)

        self.chat_logger.clear()

        if self.memory_store is not None:
            self.memory_store.clear()

        if self.opening_line_path.exists():
            opening_line = self.opening_line_path.read_text(encoding="utf-8").rstrip("\n")
            self.chat_logger.log_message(role="assistant", content=opening_line)
            logger.info("Session reset — chat log reseeded from opening_line.md")
            return "✅ Session reset — starting fresh."

        logger.info("Session reset — no opening_line.md found, chat log left empty")
        return "✅ Session reset — chat log is empty (no opening_line.md found)."
