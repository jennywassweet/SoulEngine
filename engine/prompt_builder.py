"""
Prompt builder — assembles the epistolary prompt: system frame + one user
turn containing both character bibles, the shared relationship note,
continuity notes, and the correspondence collapsed into a single manuscript.
"""

import logging
from pathlib import Path
from typing import Any

from engine.chat_logger import ChatLogger
from engine.config_utils import RUNTIME_DIR_NAME

logger = logging.getLogger(__name__)


def load_markdown_file(path: Path) -> str:
    """Read a markdown file, stripped, returning "" with a warning if missing.

    Shared with the photo engine (engine/photo_director.py), which reads
    bible/relationship/continuity the same way but outside PromptBuilder —
    not a private implementation detail of this class.
    """
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return ""
    return path.read_text(encoding="utf-8").strip()


class PromptBuilder:
    """Builds the two-message (system + user) epistolary prompt under a char budget"""

    def __init__(
        self,
        base_path: Path | str,
        setting: str,
        characters_config: dict[str, str],
        main_prompt_file: Path | str,
        budget: dict[str, int] | None = None,
        template_vars: dict[str, str] | None = None,
    ):
        """
        Initialize prompt builder

        Args:
            base_path: Base path for the project (paths are resolved relative to it)
            setting: Active setting name — everything (characters, relationship,
                     continuity) is resolved under soul/settings/<setting>/
            characters_config: {"self": "<folder>", "other": "<folder>"} under
                                soul/settings/<setting>/characters/
            main_prompt_file: Path to the system frame template
            budget: {"total_chars": int, "reserved_for_response": int}
            template_vars: Extra {{key}} substitutions available to loaded .md files
        """
        self.base_path = Path(base_path)
        self.setting_path = self.base_path / "soul" / "settings" / setting
        self.runtime_path = self.setting_path / RUNTIME_DIR_NAME
        self.main_prompt_file = Path(main_prompt_file)
        self.budget = budget or {}
        self.template_vars = template_vars or {}

        self_slug = characters_config.get("self", "self")
        other_slug = characters_config.get("other", "other")
        self.self_slug = self_slug
        self.other_slug = other_slug
        self.self_name = self_slug.capitalize()
        self.other_name = other_slug.capitalize()

        self.role_names = {
            "assistant": self.self_name,
            "user": self.other_name,
            "system": "SYSTEM",
        }

        logger.info(f"PromptBuilder initialized (self={self.self_name}, other={self.other_name})")

    def _load_markdown_file(self, path: Path) -> str:
        return load_markdown_file(path)

    def _apply_template_vars(self, text: str, extra_vars: dict[str, str]) -> str:
        all_vars = {**self.template_vars, **extra_vars}
        for key, value in all_vars.items():
            text = text.replace("{{" + key + "}}", value)
        return text

    def _format_manuscript(self, messages: list[dict[str, str]], current_message: str) -> str:
        """Collapse chat history + current incoming message into one script-style block"""
        lines = []
        for entry in messages:
            name = self.role_names.get(entry.get("role", "user"), entry.get("role", ""))
            lines.append(f"{name}: {entry.get('content', '')}")
        lines.append(f"{self.other_name}: {current_message}")
        return "\n".join(lines)

    def build_prompt(
        self,
        current_message: str,
        chat_logger: ChatLogger,
        recalled: str = "",
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """
        Build the epistolary prompt

        Args:
            current_message: Latest incoming message from `other`
            chat_logger: ChatLogger to pull manuscript history from
            recalled: Optional pre-formatted block of retrieval-memory
                excerpts (see engine/retriever.py::Retriever.format_block),
                inserted between continuity and the manuscript. Empty
                string omits the block entirely.

        Returns:
            Tuple of (messages, debug_info). messages is exactly
            [{"role": "system", ...}, {"role": "user", ...}].
        """
        name_vars = {"self_name": self.self_name, "other_name": self.other_name}

        system_content = self._apply_template_vars(
            self._load_markdown_file(self.main_prompt_file), name_vars
        )

        self_bible = self._apply_template_vars(
            self._load_markdown_file(self.setting_path / "characters" / self.self_slug / "bible.md"),
            name_vars,
        )
        other_bible = self._apply_template_vars(
            self._load_markdown_file(self.setting_path / "characters" / self.other_slug / "bible.md"),
            name_vars,
        )
        relationship = self._apply_template_vars(
            self._load_markdown_file(self.setting_path / "relationship.md"), name_vars
        )
        continuity = self._apply_template_vars(
            self._load_markdown_file(self.runtime_path / "continuity.md"), name_vars
        )

        trigger = f"Write {self.self_name}'s next message now."

        fixed_chars = (
            len(system_content)
            + len(self_bible)
            + len(other_bible)
            + len(relationship)
            + len(continuity)
            + len(recalled)
            + len(trigger)
            + len(current_message)
        )

        total_budget = self.budget.get("total_chars", 24000)
        reserved_for_response = self.budget.get("reserved_for_response", 4000)
        manuscript_budget = total_budget - fixed_chars - reserved_for_response
        if manuscript_budget < 0:
            logger.warning(
                f"Fixed content exceeds budget! Used: {fixed_chars}, Budget: {total_budget}"
            )
            manuscript_budget = 0

        history = chat_logger.read_last_n_chars(manuscript_budget)
        manuscript = self._format_manuscript(history, current_message)

        user_content = "\n\n".join(
            part
            for part in [self_bible, other_bible, relationship, continuity, recalled, manuscript, trigger]
            if part
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        total_chars = len(system_content) + len(user_content)
        debug_info = {
            "self_name": self.self_name,
            "other_name": self.other_name,
            "blocks": {
                "system": len(system_content),
                "self_bible": len(self_bible),
                "other_bible": len(other_bible),
                "relationship": len(relationship),
                "continuity": len(continuity),
                "recalled": len(recalled),
                "manuscript": len(manuscript),
            },
            "manuscript_messages": len(history),
            "current_message_chars": len(current_message),
            "total_chars": total_chars,
            "budget_limit": total_budget,
            "budget_used": (total_chars / total_budget) * 100 if total_budget > 0 else 0,
        }

        logger.info(
            f"Prompt built: {total_chars} chars ({debug_info['budget_used']:.1f}% of budget)"
        )

        return messages, debug_info
