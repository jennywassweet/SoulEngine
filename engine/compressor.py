"""
Compressor - handles chat history compression into continuity notes
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

from engine.llm.base import LLMProvider
from engine.chat_logger import ChatLogger
from engine.state_manager import StateManager
from engine.response_parser import parse_markers
from engine.debug_logger import save_prompt_to_file, append_debug_log

logger = logging.getLogger(__name__)


class Compressor:
    """Handles chat history compression into rolling continuity notes"""

    def __init__(
        self,
        llm: LLMProvider,
        chat_logger: ChatLogger,
        state_manager: StateManager,
        config: dict[str, Any],
        prompt_file: Path | str,
        summaries_path: Path | str,
        continuity_path: Path | str,
        role_names: dict[str, str] | None = None,
        template_vars: dict[str, str] | None = None,
    ):
        """
        Initialize compressor

        Args:
            llm: LLM provider
            chat_logger: ChatLogger for reading/manipulating chat log
            state_manager: StateManager for state.md (to reset counters)
            config: Compression config from soul/config.yaml
            prompt_file: Path to compression prompt template
            summaries_path: Path to summaries.jsonl (per-chunk archive log)
            continuity_path: Path to continuity.md (rolling notes, fed into the prompt)
            role_names: Mapping of role -> display name (e.g. {"user": "Misha", "assistant": "Zoya"})
            template_vars: Variables for {{key}} substitution in prompt templates
        """
        self.llm = llm
        self.chat_logger = chat_logger
        self.state_manager = state_manager
        self.config = config
        self.prompt_file = Path(prompt_file)
        self.summaries_path = Path(summaries_path)
        self.continuity_path = Path(continuity_path)
        self.role_names = role_names or {"user": "user", "assistant": "assistant"}
        self.template_vars = template_vars or {}

        # Extract config values (no magic numbers!)
        self.trigger_chars = config.get("trigger_chars", 8000)
        self.chunk_size_messages = config.get("chunk_size_messages", 20)
        self.summary_max_words = config.get("summary_max_words", 200)
        self.continuity_max_words = config.get("continuity_max_words", 570)

        # Ensure files exist
        self.summaries_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.summaries_path.exists():
            self.summaries_path.touch()

        # Created empty, not with a placeholder word: whatever is in this
        # file goes into the prompt verbatim, and PromptBuilder drops empty
        # parts - so before the first compression the continuity block is
        # simply absent rather than saying "nothing here yet" in the
        # character's voice.
        self.continuity_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.continuity_path.exists():
            self.continuity_path.touch()

        logger.info(
            f"Compressor initialized (trigger: {self.trigger_chars} chars, "
            f"chunk: {self.chunk_size_messages} messages, "
            f"limits: summary={self.summary_max_words} words, continuity={self.continuity_max_words} words)"
        )

    def _load_prompt_template(self) -> str:
        """Load prompt template and replace placeholders"""
        if not self.prompt_file.exists():
            logger.error(f"Compression prompt file not found: {self.prompt_file}")
            return ""

        template = self.prompt_file.read_text(encoding="utf-8")

        # Build all variables: config values + template_vars
        all_vars = {
            "summary_max_words": str(self.summary_max_words),
            "continuity_max_words": str(self.continuity_max_words),
            **self.template_vars,
        }

        for key, value in all_vars.items():
            template = template.replace("{{" + key + "}}", value)

        return template

    def _count_words(self, text: str) -> int:
        """Count words in text (simple split by whitespace)"""
        return len(text.split())

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for compression"""
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            name = self.role_names.get(role, role)
            content = msg.get("content", "")
            lines.append(f"[{name}] {content}")
        return "\n".join(lines)

    def _read_continuity(self) -> str:
        """Read current continuity notes"""
        if not self.continuity_path.exists():
            return ""

        return self.continuity_path.read_text(encoding="utf-8").strip()

    def _write_continuity(self, continuity: str) -> None:
        """Write continuity notes to file"""
        self.continuity_path.write_text(continuity.strip() + "\n", encoding="utf-8")

    def _append_summary(self, summary: str, messages_count: int) -> None:
        """Append summary to summaries.jsonl"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "messages_compressed": messages_count,
            "char_count": len(summary),
        }

        with open(self.summaries_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _delete_old_messages(self, count: int) -> None:
        """Delete first N messages from chat_log.jsonl"""
        all_messages = self.chat_logger.read_all()
        self.chat_logger.rewrite(all_messages[count:])

    async def compress(self) -> dict[str, Any]:
        """
        Compress oldest messages in chat log

        Returns:
            Dictionary with result:
            - success: bool
            - messages_compressed: int (if success)
            - summary_chars: int (if success)
            - continuity_chars: int (if success)
            - chat_log_remaining: int (if success)
            - error: str (if not success)
            - raw_response: str (if parse failed)
        """
        logger.info("Compression started")

        try:
            # Load prompt template
            prompt_template = self._load_prompt_template()
            if not prompt_template:
                return {
                    "success": False,
                    "error": "prompt_template_missing"
                }

            # Read current continuity notes
            current_continuity = self._read_continuity()

            # Get oldest messages to compress
            all_messages = self.chat_logger.read_all()
            messages_to_compress = all_messages[:self.chunk_size_messages]

            if not messages_to_compress:
                logger.warning("No messages to compress")
                return {
                    "success": False,
                    "error": "no_messages"
                }

            formatted_messages = self._format_messages(messages_to_compress)

            # Build context for LLM
            context = (
                f"CONTINUITY:\n{current_continuity if current_continuity else '(none yet)'}\n\n"
                f"MESSAGES:\n{formatted_messages}"
            )

            # Call LLM
            messages = [
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": context},
            ]

            # Save prompt for debugging
            save_prompt_to_file(messages, Path("debug/last_compression_prompt.txt"), self.role_names)

            logger.debug("Calling LLM for compression...")
            response = await self.llm.complete(
                messages=messages,
                temperature=0.7,
                max_tokens=4000,
            )

            raw_response = response.content
            llm_chars = len(raw_response)

            logger.debug(f"LLM compression response: {llm_chars} chars")

            # Parse response
            parsed = parse_markers(raw_response, ["===SUMMARY===", "===CONTINUITY==="])

            new_summary = parsed.get("SUMMARY")
            new_continuity = parsed.get("CONTINUITY")

            # Check parsing success
            if new_summary is None or new_continuity is None:
                logger.error("Compression parse failed. Markers not found in response.")
                logger.error(f"Raw response preview: {raw_response[:500]}")
                return {
                    "success": False,
                    "error": "parse_failed",
                    "raw_response": raw_response,
                }

            # Count words
            summary_words = self._count_words(new_summary)
            continuity_words = self._count_words(new_continuity)

            # Trim to word limits
            if summary_words > self.summary_max_words:
                logger.warning(
                    f"SUMMARY exceeds limit ({summary_words} > {self.summary_max_words} words), trimming"
                )
                words = new_summary.split()
                new_summary = " ".join(words[:self.summary_max_words])

            if continuity_words > self.continuity_max_words:
                logger.warning(
                    f"CONTINUITY exceeds limit ({continuity_words} > {self.continuity_max_words} words), trimming"
                )
                words = new_continuity.split()
                new_continuity = " ".join(words[:self.continuity_max_words])

            # Save results
            self._append_summary(new_summary, len(messages_to_compress))
            self._write_continuity(new_continuity)

            append_debug_log(
                {
                    "type": "compression",
                    "messages": messages,
                    "raw_response": raw_response,
                    "summary": new_summary,
                    "continuity": new_continuity,
                    "messages_compressed": len(messages_to_compress),
                },
                Path("debug/prompt_log.jsonl"),
            )

            # Delete compressed messages from chat log
            self._delete_old_messages(len(messages_to_compress))

            # Reset compression counter
            self.state_manager.reset_counter("total_chars_since_last_compression")

            # Count remaining messages
            remaining_messages = len(self.chat_logger.read_all())

            # Recount after trimming
            final_summary_words = self._count_words(new_summary)
            final_continuity_words = self._count_words(new_continuity)

            logger.info(
                f"Compression completed: compressed {len(messages_to_compress)} messages, "
                f"summary={final_summary_words} words, continuity={final_continuity_words} words, "
                f"remaining={remaining_messages} messages"
            )

            return {
                "success": True,
                "messages_compressed": len(messages_to_compress),
                "summary_words": final_summary_words,
                "continuity_words": final_continuity_words,
                "summary_chars": len(new_summary),
                "continuity_chars": len(new_continuity),
                "chat_log_remaining": remaining_messages,
                "summary": new_summary,
                "continuity": new_continuity,
            }

        except Exception as e:
            logger.error(f"Compression failed with exception: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"exception: {str(e)}"
            }

    def should_compress(self) -> bool:
        """
        Check if compression should be triggered

        Returns:
            True if compression should run
        """
        frontmatter = self.state_manager.get_frontmatter()
        chars_since_compression = frontmatter.get("total_chars_since_last_compression", 0)

        if chars_since_compression >= self.trigger_chars:
            logger.debug(
                f"Compression triggered: {chars_since_compression} >= {self.trigger_chars} chars"
            )
            return True

        return False
