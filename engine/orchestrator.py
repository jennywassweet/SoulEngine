"""
Orchestrator - connects LLM brain with external connectors
"""

import logging
import os
import sys
from collections import deque
from pathlib import Path
from datetime import datetime

from engine.llm.base import LLMProvider
from engine.llm.mistral import MistralProvider
from engine.llm.openrouter import OpenRouterProvider
from engine.connectors.base import Connector, IncomingMessage
from engine.prompt_builder import PromptBuilder, load_markdown_file
from engine.chat_logger import ChatLogger
from engine.state_manager import StateManager
from engine.compressor import Compressor
from engine.command_parser import parse_command
from engine.command_executor import CommandExecutor
from engine.config_utils import resolve_images_config, runtime_dir, setting_dir
from engine.debug_logger import save_prompt_to_file, append_debug_log
from engine.embeddings.openrouter import OpenRouterEmbeddingProvider
from engine.images.openrouter import OpenRouterImageProvider
from engine.memory_store import MemoryStore
from engine.photo_director import PhotoDirector, PhotoModeUnavailable
from engine.photo_store import PhotoStore
from engine.retriever import Retriever

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDERS: dict[str, type] = {
    "openrouter": OpenRouterEmbeddingProvider,
}

IMAGE_PROVIDERS: dict[str, type] = {
    "openrouter": OpenRouterImageProvider,
}

# Photo mode's describer is a second, independent text-model call - its
# own small registry rather than importing
# main.py's LLM_PROVIDERS, which would create a circular import (main.py
# constructs Orchestrator). Mirrors EMBEDDING_PROVIDERS above.
DESCRIBER_LLM_PROVIDERS: dict[str, type] = {
    "mistral": MistralProvider,
    "openrouter": OpenRouterProvider,
}

# Shown by /help and by the unknown-command fallback. Covers both dispatchers:
# control commands (command_parser.py -> CommandExecutor) and the debug
# commands below — they're deliberately separate code paths, but from inside
# the chat they're one flat namespace, so listing only half of them hides the
# commands that actually change state.
HELP_TEXT = (
    "📖 **Commands**\n"
    "\n**Control** (change state):\n"
    "/list - List available settings\n"
    "/switch <setting> - Switch to another setting (restarts the bot)\n"
    "/reset - Wipe this setting's history and start fresh\n"
    "/back N - Roll back the last N exchanges\n"
    "\n**Where were we**:\n"
    "/recap - Who you're writing to, and where the story stands\n"
    "\n**Debug**:\n"
    "/debug - Show prompt composition\n"
    "/state - Show current state\n"
    "/user - Show user model\n"
    "/config - Show config\n"
    "/memory - Show retrieval memory state\n"
    "/compress - Force compression now\n"
    "/regen - Regenerate the last reply\n"
    "/reset_counters - Reset compression counters (does not touch chat history)\n"
    "\n**Photo** (separate from the text circuit):\n"
    "/shot [hint] - Her phone, her point of view (no face)\n"
    "/scene [hint] - An unseen observer in the room (face + world)\n"
    "/selfie [hint] - Her phone, pointed at herself (face required)\n"
    "/again [modifier] - Revise and resend the last delivered frame\n"
    "/photos [n] - Show the last n photo requests (default 5)"
)

# Names PHOTO_COMMANDS covers must have a "/name" line in HELP_TEXT above —
# see tests/test_help_text.py.
PHOTO_COMMANDS = {"shot", "scene", "selfie", "again", "photos"}

# /recap falls back to this many manuscript messages when a setting is too
# young to have continuity notes, and never sends more than this many
# characters — Telegram's own hard limit is 4096.
RECAP_FALLBACK_MESSAGES = 6
RECAP_MAX_CHARS = 3500


class Orchestrator:
    """
    Main orchestrator that connects LLM with connectors.
    Handles message routing, response generation, and state management.
    """

    def __init__(
        self,
        llm: LLMProvider,
        connector: Connector,
        base_path: Path | str,
        config: dict,  # Full config from soul/config.yaml
        chat_log_path: Path | str,
        state_path: Path | str,
        user_model_path: Path | str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        reasoning_effort: str | None = None,
        provider_routing: dict | None = None,
    ):
        """
        Initialize orchestrator

        Args:
            llm: LLM provider instance
            connector: Communication connector instance
            base_path: Base path for the project
            config: Full configuration dictionary from soul/config.yaml
            chat_log_path: Path to chat_log.jsonl
            state_path: Path to state.md
            user_model_path: Path to user_model.md
            temperature: LLM temperature parameter
            max_tokens: Max tokens for LLM response
            reasoning_effort: Optional reasoning effort ("low"/"medium"/"high") for
                models with mandatory reasoning
            provider_routing: Optional OpenRouter `provider` object (order/only/
                ignore/allow_fallbacks/etc.), ignored by providers without an
                equivalent concept
        """
        self.llm = llm
        self.connector = connector
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.provider_routing = provider_routing
        self.config = config

        # Initialize components
        self.base_path = Path(base_path)

        setting = config.get("setting", "default")
        setting_path = setting_dir(self.base_path, setting)
        runtime_path = runtime_dir(self.base_path, setting)
        characters_config = config.get("characters", {})
        prompt_config = config.get("prompt", {})

        def _resolve_prompt_file(override_name: str, config_default: Path) -> Path:
            """Per-setting prompt file if present, else the shared config default."""
            override = setting_path / override_name
            return override if override.exists() else config_default

        main_prompt_file = _resolve_prompt_file(
            "main_prompt.md",
            self.base_path / prompt_config.get("main_prompt_file", "config/main_prompt.md"),
        )

        self.prompt_builder = PromptBuilder(
            base_path=base_path,
            setting=setting,
            characters_config=characters_config,
            main_prompt_file=main_prompt_file,
            budget=prompt_config.get("budget", {}),
        )
        self.role_names = self.prompt_builder.role_names
        self.self_name = self.prompt_builder.self_name
        self.other_name = self.prompt_builder.other_name
        self.template_vars = {"self_name": self.self_name, "other_name": self.other_name}

        self.state_manager = StateManager(state_path)
        self.user_model_manager = StateManager(user_model_path)
        self.chat_logger = ChatLogger(chat_log_path, state_manager=self.state_manager)

        # Initialize compressor
        compression_config = config.get("compression", {})
        compression_prompt_file = _resolve_prompt_file(
            "compression_prompt.md",
            self.base_path / compression_config.get("prompt_file", "config/compression_prompt.md"),
        )
        summaries_path = runtime_path / "summaries.jsonl"
        continuity_path = runtime_path / "continuity.md"

        self.compressor = Compressor(
            llm=llm,
            chat_logger=self.chat_logger,
            state_manager=self.state_manager,
            config=compression_config,
            prompt_file=compression_prompt_file,
            summaries_path=summaries_path,
            continuity_path=continuity_path,
            role_names=self.role_names,
            template_vars=self.template_vars,
        )

        # Retrieval memory. Disabled by
        # default; all four stay None/unset and the rest of this class
        # treats that as "feature off".
        memory_config = config.get("memory", {})
        self.embedding_provider = None
        self.memory_store = None
        self.retriever = None
        self.recalled_header = None
        # message_ids recalled in the last N turns, held out of the next
        # searches. Measured need: without it one topical cluster kept
        # winning and was re-injected into ~40% of turns, which is how the
        # continuity repetition loop starts. In memory only — a restart
        # forgets it, and the worst that costs is one repeated fragment.
        self.recent_recalls: deque[list[int]] = deque(maxlen=0)

        if memory_config.get("enabled", False):
            provider_name = memory_config.get("provider", "openrouter")
            provider_cls = EMBEDDING_PROVIDERS.get(provider_name)
            if provider_cls is None:
                raise ValueError(
                    f"Unknown memory.provider '{provider_name}' — must be one of {list(EMBEDDING_PROVIDERS)}"
                )
            embedding_kwargs = {}
            if memory_config.get("model"):
                embedding_kwargs["model"] = memory_config["model"]
            self.embedding_provider = provider_cls(**embedding_kwargs)

            self.memory_store = MemoryStore(
                jsonl_path=runtime_path / "memory.jsonl",
                npy_path=runtime_path / "memory_vectors.npy",
                model_name=self.embedding_provider.model_name,
            )
            self.retriever = Retriever(
                store=self.memory_store,
                embedding_provider=self.embedding_provider,
                config=memory_config,
                role_names=self.role_names,
            )
            self.recalled_header = memory_config.get(
                "recalled_header", "Fragments from earlier in the exchange:"
            )
            self.recent_recalls = deque(maxlen=memory_config.get("recall_cooldown_turns", 3))
            logger.info(f"Retrieval memory enabled (provider={provider_name}, {len(self.memory_store)} messages archived)")

        # Photo mode — a fully separate engine:
        # its own store, its own describer LLM call, read-only access to
        # bible/relationship/continuity. Nothing here touches chat_logger,
        # memory, or continuity. self.photo_store/self.photo_director stay
        # None unless this setting has soul/settings/<setting>/images.yaml
        # declaring at least one mode; PhotoDirector's own `enabled` check
        # (from the global images.enabled switch) governs actual calls, so
        # /photos and /again can still read/reuse old frames while the
        # global switch is off.
        images_config = resolve_images_config(self.base_path, setting, config.get("images", {}))
        self.photo_store = None
        self.photo_director = None
        if images_config.get("modes"):
            image_provider_name = images_config.get("provider", "openrouter")
            image_provider_cls = IMAGE_PROVIDERS.get(image_provider_name)
            if image_provider_cls is None:
                raise ValueError(
                    f"Unknown images.provider '{image_provider_name}' — must be one of {list(IMAGE_PROVIDERS)}"
                )
            image_provider_kwargs = {}
            if images_config.get("model"):
                image_provider_kwargs["model"] = images_config["model"]
            if images_config.get("max_references") is not None:
                image_provider_kwargs["max_references"] = images_config["max_references"]
            image_provider = image_provider_cls(**image_provider_kwargs)

            describer_config = images_config.get("llm", {})
            describer_provider_name = describer_config.get("provider", "openrouter")
            describer_provider_cls = DESCRIBER_LLM_PROVIDERS.get(describer_provider_name)
            if describer_provider_cls is None:
                raise ValueError(
                    f"Unknown images.llm.provider '{describer_provider_name}' — "
                    f"must be one of {list(DESCRIBER_LLM_PROVIDERS)}"
                )
            describer_kwargs = {}
            if describer_config.get("model"):
                describer_kwargs["model"] = describer_config["model"]
            describer_llm = describer_provider_cls(**describer_kwargs)

            photo_prompt_file = _resolve_prompt_file(
                "photo_prompt.md",
                self.base_path / images_config.get("prompt_file", "config/photo_prompt.md"),
            )
            self.photo_store = PhotoStore(
                photos_dir=runtime_path / "photos",
                jsonl_path=runtime_path / "photos.jsonl",
            )
            self.photo_director = PhotoDirector(
                llm=describer_llm,
                image_provider=image_provider,
                photo_store=self.photo_store,
                chat_logger=self.chat_logger,
                setting_path=setting_path,
                self_slug=self.prompt_builder.self_slug,
                role_names=self.role_names,
                config=images_config,
                prompt_file=photo_prompt_file,
                describer_model=describer_config.get("model", getattr(describer_llm, "model", "default")),
                template_vars=self.template_vars,
                debug_prompt_path=self.base_path / "debug" / "last_photo_prompt.txt",
            )
            logger.info(
                f"Photo mode enabled (modes={images_config['modes']}, "
                f"enabled={images_config.get('enabled', False)}, image_model={image_provider.model_name})"
            )

        # Control commands (e.g. /reset) are parsed and executed before
        # anything reaches the LLM path or chat_log.jsonl — see
        # command_parser.py / command_executor.py.
        self.command_executor = CommandExecutor(
            chat_logger=self.chat_logger,
            state_path=state_path,
            user_model_path=user_model_path,
            continuity_path=continuity_path,
            summaries_path=summaries_path,
            opening_line_path=setting_path / "opening_line.md",
            debug_dir=self.base_path / "debug",
            memory_store=self.memory_store,
            base_path=self.base_path,
            setting=setting,
        )

        # Store last debug info for /debug command
        self.last_debug_info = None
        # Entries (with "score") returned by the last retrieve_with_vector()
        # call, for the /memory debug command. [] when memory is disabled,
        # retrieval failed, or nothing matched.
        self.last_recalled: list[dict] = []

        # Register message handler
        self.connector.on_message(self._handle_message)

        logger.info("Orchestrator initialized with prompt assembly + compression")

    def _is_debug_command(self, text: str) -> bool:
        """Check if message is a debug command"""
        return text.strip().startswith("/")

    async def _handle_debug_command(self, command: str, message: IncomingMessage) -> None:
        """
        Handle debug commands

        Args:
            command: Command text (e.g., "/debug", "/state", "/shot a hint")
            message: Incoming message
        """
        # Name and args are split apart before comparison — only the name is
        # lowercased. A viewer's photo hint (/shot <hint>) may be in Russian
        # and its case is meaningful, and command args in general were never
        # meant to be case-folded; comparing the whole raw string (as this
        # used to) meant any debug command with arguments could never match.
        stripped = command.strip()
        name, _, args = stripped.partition(" ")
        name = name.lower()
        args = args.strip()

        if name in ("/help", "/start"):
            await self.connector.send(message.channel_id, HELP_TEXT)
        elif name == "/debug":
            await self._cmd_debug(message)
        elif name == "/recap":
            await self._cmd_recap(message)
        elif name == "/state":
            await self._cmd_state(message)
        elif name == "/user":
            await self._cmd_user(message)
        elif name == "/reset_counters":
            await self._cmd_reset_counters(message)
        elif name == "/compress":
            await self._cmd_compress(message)
        elif name == "/config":
            await self._cmd_config(message)
        elif name == "/memory":
            await self._cmd_memory(message)
        elif name == "/regen":
            await self._cmd_regen(message)
        elif name in ("/shot", "/scene", "/selfie"):
            await self._cmd_photo(name[1:], args, message)
        elif name == "/again":
            await self._cmd_photo_again(args, message)
        elif name == "/photos":
            await self._cmd_photos(args, message)
        else:
            await self.connector.send(
                message.channel_id,
                f"Unknown command: {command}\n\n" + HELP_TEXT
            )

    async def _cmd_recap(self, message: IncomingMessage) -> None:
        """Who the character is and where the story stands.

        Not a debug command: the chat window is a view, not storage. Clearing
        it, or switching to another setting and back, costs the visible thread
        while chat_log.jsonl keeps everything — this prints the same continuity
        notes the model itself is working from, so the reader picks the story
        back up where the character already is.
        """
        continuity = load_markdown_file(self.compressor.continuity_path)

        if continuity:
            body = continuity
        else:
            # Continuity notes don't exist until the first compression, so a
            # young setting would get an empty answer to a fair question. Fall
            # back to the tail of the manuscript, formatted the way the
            # manuscript itself reads.
            history = self.chat_logger.read_all()[-RECAP_FALLBACK_MESSAGES:]
            if not history:
                await self.connector.send(
                    message.channel_id,
                    f"**{self.self_name}** — nothing yet, this story hasn't started.",
                )
                return
            body = "\n".join(
                f"{self.role_names.get(entry.get('role'), entry.get('role'))}: "
                f"{entry.get('content', '')}"
                for entry in history
            )

        # Telegram rejects anything over 4096 characters outright; continuity
        # is bounded by compression.continuity_max_words but a manuscript tail
        # is not, so cut rather than lose the whole message to an API error.
        if len(body) > RECAP_MAX_CHARS:
            body = body[:RECAP_MAX_CHARS].rstrip() + " […]"

        await self.connector.send(
            message.channel_id,
            f"📖 **{self.self_name}** — where things stand\n\n{body}",
        )

    async def _cmd_debug(self, message: IncomingMessage) -> None:
        """Show prompt composition debug info"""
        if not self.last_debug_info:
            await self.connector.send(
                message.channel_id,
                "No debug info available yet. Send a message first."
            )
            return

        info = self.last_debug_info
        lines = [
            "🔍 **Prompt Composition Debug**\n",
            f"**Budget**: {info['total_chars']}/{info['budget_limit']} chars ({info['budget_used']:.1f}%)\n",
            f"**Current message**: {info['current_message_chars']} chars\n",
            "\n**Blocks**:",
        ]

        for name, chars in info["blocks"].items():
            lines.append(f"  • {name}: {chars} chars")

        await self.connector.send(message.channel_id, "\n".join(lines))

    async def _cmd_state(self, message: IncomingMessage) -> None:
        """Show current state"""
        frontmatter = self.state_manager.get_frontmatter()

        lines = [
            "📊 **Current State**\n",
            f"**Session messages**: {frontmatter.get('session_message_count', 0)}",
            f"**Chars since compression**: {frontmatter.get('total_chars_since_last_compression', 0)}",
            f"**Last updated**: {frontmatter.get('last_updated', 'never')}",
        ]

        await self.connector.send(message.channel_id, "\n".join(lines))

    async def _cmd_user(self, message: IncomingMessage) -> None:
        """Show user model"""
        frontmatter = self.user_model_manager.get_frontmatter()
        content = self.user_model_manager.get_content()

        lines = [
            "👤 **User Model**\n",
            f"**User ID**: {frontmatter.get('user_id', 'unknown')}",
            f"**First seen**: {frontmatter.get('first_seen', 'unknown')}",
            f"**Last seen**: {frontmatter.get('last_seen', 'unknown')}",
            f"**Interactions**: {frontmatter.get('interaction_count', 0)}",
        ]

        if content:
            lines.append(f"\n**Notes**:\n{content}")

        await self.connector.send(message.channel_id, "\n".join(lines))

    async def _cmd_reset_counters(self, message: IncomingMessage) -> None:
        """Debug-only: zero the compression bookkeeping counters in state.md.

        Does not touch chat_log.jsonl/continuity.md — for a real "start this
        conversation over" reset, use the /reset control command instead
        (handled earlier in _handle_message, via CommandExecutor).
        """
        self.state_manager.update_frontmatter({
            "session_message_count": 0,
            "total_chars_since_last_compression": 0,
            "last_updated": datetime.now().isoformat(),
        })

        await self.connector.send(
            message.channel_id,
            "✅ Counters reset successfully"
        )

    async def _cmd_compress(self, message: IncomingMessage) -> None:
        """Force compression"""
        await self.connector.send(message.channel_id, "📦 Running compression...")

        result = await self.compressor.compress()

        if result.get("success"):
            response = (
                "📦 **Compression Result**\n\n"
                f"Status: ✅ success\n"
                f"Messages compressed: {result['messages_compressed']}\n"
                f"Summary: {result.get('summary_words', 0)} words "
                f"(limit: {self.compressor.summary_max_words})\n"
                f"Continuity: {result.get('continuity_words', 0)} words "
                f"(limit: {self.compressor.continuity_max_words})\n"
                f"Chat log remaining: {result['chat_log_remaining']} messages\n\n"
                f"--- SUMMARY ---\n{result.get('summary', '')}\n\n"
                f"--- CONTINUITY ---\n{result.get('continuity', '')}"
            )
        else:
            error = result.get("error", "unknown")
            response = (
                "📦 **Compression Result**\n\n"
                f"Status: ❌ failed\n"
                f"Error: {error}"
            )
            if "raw_response" in result:
                response += f"\n\nRaw response preview:\n{result['raw_response'][:200]}"

        await self.connector.send(message.channel_id, response)

    async def _cmd_config(self, message: IncomingMessage) -> None:
        """Show current config"""
        compression_config = self.config.get("compression", {})
        frontmatter = self.state_manager.get_frontmatter()

        response = (
            "⚙️ **Current Config**\n\n"
            "**Compression:**\n"
            f"  Trigger: {compression_config.get('trigger_chars', 'N/A')} chars\n"
            f"  Chunk size: {compression_config.get('chunk_size_messages', 'N/A')} messages\n"
            f"  Summary limit: {compression_config.get('summary_max_words', 'N/A')} words\n"
            f"  Continuity limit: {compression_config.get('continuity_max_words', 'N/A')} words\n"
            "\n**Counters:**\n"
            f"  Messages this session: {frontmatter.get('session_message_count', 0)}\n"
            f"  Chars since last compression: {frontmatter.get('total_chars_since_last_compression', 0)}"
        )

        await self.connector.send(message.channel_id, response)

    async def _cmd_memory(self, message: IncomingMessage) -> None:
        """Debug-only: retrieval memory store size/model + what got recalled last turn."""
        if self.memory_store is None:
            await self.connector.send(
                message.channel_id, "Retrieval memory is disabled (memory.enabled: false)."
            )
            return

        lines = [
            "🧠 **Retrieval Memory**\n",
            f"**Model**: {self.embedding_provider.model_name}",
            f"**Archived messages**: {len(self.memory_store)}",
            "\n**Recalled last turn**:",
        ]

        if not self.last_recalled:
            lines.append("  (nothing)")
        else:
            for entry in self.last_recalled:
                score = entry.get("score")
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
                preview = entry.get("content", "")[:60]
                lines.append(f"  • id={entry.get('message_id')} score={score_str} — {preview}")

        await self.connector.send(message.channel_id, "\n".join(lines))

    async def _cmd_regen(self, message: IncomingMessage) -> None:
        """
        Regenerate the reply to the current last user message: drop the
        trailing assistant reply (if any, along with its retrieval-memory
        vector), then generate a fresh one.

        Also serves as retry-after-failure recovery, with no separate code
        path: if the tail is already a dangling unanswered user message
        (a prior turn's LLM call failed after logging it), there's no
        assistant reply to drop, so this just generates the missing one.
        """
        all_messages = self.chat_logger.read_all()

        if all_messages and all_messages[-1].get("role") == "assistant":
            dropped = all_messages.pop()
            if self.memory_store is not None and dropped.get("message_id") is not None:
                self.memory_store.delete_from(dropped["message_id"])

        if not all_messages or all_messages[-1].get("role") != "user":
            self.chat_logger.rewrite(all_messages)
            await self.connector.send(
                message.channel_id, "Nothing to regenerate — no pending user message."
            )
            return

        # Pull the user message out too: build_prompt() reads history from
        # chat_logger AND appends current_message itself, so it must not
        # already be the last entry on disk (same rule _handle_message
        # follows for a normal incoming message).
        last_user = all_messages[-1]
        self.chat_logger.rewrite(all_messages[:-1])

        try:
            await self._generate_and_send(
                incoming_text=last_user.get("content", ""),
                channel_id=message.channel_id,
                incoming_user_id=last_user.get("user_id"),
                incoming_message_id=last_user.get("message_id"),
                archive_incoming=False,  # already archived under this same id, first time around
            )
        except Exception:
            # Don't lose the user's message if generation blew up before
            # re-logging it (_generate_and_send re-logs it early, so this
            # only matters for a failure before that point).
            self.chat_logger.rewrite(all_messages)
            raise

    async def _cmd_photo(self, mode: str, hint: str, message: IncomingMessage) -> None:
        """/shot, /scene, /selfie — request one frame. Fully isolated from
        the text circuit: no chat_log/memory/continuity read or write
        happens here. Never lets a photo-engine
        failure reach the user as anything but a plain text remark, and
        never lets one propagate up to break the bot."""
        if self.photo_director is None:
            await self.connector.send(message.channel_id, "Photo mode isn't set up for this setting.")
            return

        await self._send_chat_action_safely(message.channel_id, "upload_photo")

        try:
            entry = await self.photo_director.generate(mode=mode, hint=hint)
        except PhotoModeUnavailable as e:
            await self.connector.send(message.channel_id, str(e))
            return
        except Exception as e:
            logger.error(f"Photo generation failed (mode={mode}): {e}", exc_info=True)
            await self.connector.send(message.channel_id, "Couldn't get a frame this time.")
            return

        await self._deliver_photo_entry(entry, message.channel_id)

    async def _cmd_photo_again(self, modifier: str, message: IncomingMessage) -> None:
        """/again — resubmit the last delivered frame's prompt, optionally
        revised by `modifier`. The only continuity between frames, and it
        stays outside the fiction (see the plan)."""
        if self.photo_director is None:
            await self.connector.send(message.channel_id, "Photo mode isn't set up for this setting.")
            return

        await self._send_chat_action_safely(message.channel_id, "upload_photo")

        try:
            entry = await self.photo_director.again(modifier)
        except PhotoModeUnavailable as e:
            await self.connector.send(message.channel_id, str(e))
            return
        except Exception as e:
            logger.error(f"Photo regeneration failed: {e}", exc_info=True)
            await self.connector.send(message.channel_id, "Couldn't get a frame this time.")
            return

        if entry is None:
            await self.connector.send(
                message.channel_id, "Nothing to repeat yet — request a frame first (/shot, /scene, /selfie)."
            )
            return

        await self._deliver_photo_entry(entry, message.channel_id)

    async def _cmd_photos(self, args: str, message: IncomingMessage) -> None:
        """/photos [n] — the last n requests (default 5), decision and
        prompt included. This is the archive the feature is actually
        judged by, not the images themselves (see PhotoStore)."""
        if self.photo_store is None:
            await self.connector.send(message.channel_id, "Photo mode isn't set up for this setting.")
            return

        count = int(args) if args.isdigit() else 5
        entries = self.photo_store.read_last_n(count)
        if not entries:
            await self.connector.send(message.channel_id, "No photo requests yet.")
            return

        lines = ["🖼 **Recent photo requests**\n"]
        for entry in entries:
            status = "✅" if entry["available"] else "🚫"
            detail = entry["prompt"] if entry["available"] else entry["note"]
            lines.append(f"{status} #{entry['id']} [{entry['mode']}] {detail}")

        await self.connector.send(message.channel_id, "\n".join(lines))

    async def _deliver_photo_entry(self, entry: dict, channel_id: str) -> None:
        """Send a PhotoStore record's outcome: the image file if one was
        produced, otherwise its note as a plain text remark — never a
        caption on the photo. A frame arrives silently: a caption would be
        the character speaking about a picture she never sent. The prompt
        behind a frame is inspected with /photos instead."""
        if entry.get("file"):
            image_bytes = (self.photo_store.photos_dir / entry["file"]).read_bytes()
            try:
                await self.connector.send_photo(channel_id, image_bytes)
            except Exception as e:
                logger.error(f"Failed to send photo #{entry['id']}: {e}", exc_info=True)
                await self.connector.send(channel_id, "Got a frame but couldn't send it.")
        else:
            await self.connector.send(channel_id, entry.get("note") or "No frame right now.")

    async def _send_chat_action_safely(self, channel_id: str, action: str) -> None:
        """Best-effort "uploading photo" indicator — purely cosmetic, so a
        failure here must never interrupt the actual generation/send."""
        try:
            await self.connector.send_chat_action(channel_id, action)
        except Exception:
            logger.warning(f"Failed to send chat action '{action}'", exc_info=True)

    async def _remember(
        self,
        message_id: int | None,
        role: str,
        content: str,
        vector: list[float] | None = None,
    ) -> None:
        """
        Embed and archive one message into the retrieval memory store.

        Retrieval memory is an enhancement, never something that should
        block a turn: any failure here is logged and swallowed.

        Args:
            vector: Pre-computed embedding, if the caller already has one
                (e.g. batched together with the retrieval query embedding
                in _handle_message) — saves a redundant API call. Embedded
                fresh here if omitted.
        """
        if self.memory_store is None or message_id is None:
            return

        try:
            if vector is None:
                [vector] = await self.embedding_provider.embed([content])
            self.memory_store.add(
                entries=[{
                    "message_id": message_id,
                    "timestamp": datetime.now().isoformat(),
                    "role": role,
                    "content": content,
                }],
                vectors=[vector],
            )
        except Exception:
            logger.warning(f"Failed to archive message {message_id} into retrieval memory", exc_info=True)

    async def _generate_and_send(
        self,
        incoming_text: str,
        channel_id: str,
        incoming_user_id: str | None = None,
        incoming_message_id: int | None = None,
        archive_incoming: bool = True,
    ) -> None:
        """
        Shared generation pipeline: retrieval -> build prompt -> log the
        incoming message -> call the LLM -> parse -> log + archive the
        reply -> maybe compress -> send.

        Used both for a normal incoming Telegram message and for /regen,
        which re-runs this for a user message that's already in the chat
        log/memory store — hence incoming_message_id (reuse its existing id
        instead of auto-assigning a new one) and archive_incoming=False
        (don't re-embed/re-store something already archived).

        Args:
            incoming_text: The message to generate a reply to
            channel_id: Where to send the reply
            incoming_user_id: Telegram user id, if any (for logging)
            incoming_message_id: Reuse this id instead of auto-assigning one
            archive_incoming: Whether to embed+store incoming_text into
                retrieval memory (skip for /regen — it's already archived)
        """
        # Retrieval memory: search happens BEFORE build_prompt/logging, for
        # the same reason build_prompt does — the incoming message isn't in
        # chat_log.jsonl yet, so exclude_ids below (everything still "on
        # screen" in the live manuscript) can't accidentally include it.
        # A failure here must not block the turn - retrieval is an
        # enhancement, so embedding/search errors are logged and the turn
        # goes on without recall.
        recalled = ""
        recalled_ids: list[int] = []
        recalled_scores: list[float] = []
        incoming_vector: list[float] | None = None
        self.last_recalled = []

        if self.retriever is not None:
            try:
                exclude_ids = {
                    entry["message_id"]
                    for entry in self.chat_logger.read_all()
                    if entry.get("message_id") is not None
                }
                # Hold back whatever the last few turns already recalled, so
                # one strong cluster can't reappear turn after turn.
                exclude_ids |= {mid for turn in self.recent_recalls for mid in turn}
                if incoming_message_id is not None:
                    # /regen temporarily pulls the message being answered out
                    # of chat_log.jsonl (see _cmd_regen) so build_prompt()
                    # doesn't duplicate it in the manuscript — but that same
                    # gap would let it recall itself here, since it's still
                    # sitting in memory_store from when it first arrived.
                    exclude_ids.add(incoming_message_id)
                recent_history = self.chat_logger.read_last_n(self.retriever.query_window_messages)
                query_text = self.retriever.build_query_text(recent_history, incoming_text)

                # Batch the query embedding with the incoming message's own
                # embedding (needed below for archival) in one call.
                query_vector, incoming_vector = await self.embedding_provider.embed(
                    [query_text, incoming_text]
                )
                recalled_entries = await self.retriever.retrieve_with_vector(
                    query_vector, exclude_ids=exclude_ids
                )
                recalled = self.retriever.format_block(recalled_entries, header=self.recalled_header)
                recalled_ids = [entry["message_id"] for entry in recalled_entries]
                recalled_scores = [entry.get("score") for entry in recalled_entries]
                self.last_recalled = recalled_entries
                self.recent_recalls.append(recalled_ids)
            except Exception:
                logger.warning(
                    "Retrieval failed for this turn, continuing without recalled context",
                    exc_info=True,
                )

        # Build prompt using PromptBuilder. Manuscript history must reflect
        # the state BEFORE this turn — current_message is appended once,
        # explicitly, inside build_prompt. Log it to chat_logger only after.
        messages, debug_info = self.prompt_builder.build_prompt(
            current_message=incoming_text,
            chat_logger=self.chat_logger,
            recalled=recalled,
        )
        self.last_debug_info = debug_info

        # Blind-spot check: exclude_ids covers everything in chat_log.jsonl,
        # but only the tail that fits the char budget reaches the manuscript.
        # Anything in between is in neither place — invisible to the model.
        # That gap is empty as long as the manuscript budget stays larger
        # than compression.trigger_chars; warn rather than silently drop, so
        # a budget/trigger change that breaks the invariant is noticeable.
        if self.retriever is not None:
            hidden = len(self.chat_logger.read_all()) - debug_info.get("manuscript_messages", 0)
            if hidden > 0:
                logger.warning(
                    f"{hidden} logged message(s) fit neither the manuscript (char budget) nor "
                    f"retrieval (excluded as still-live) — raise prompt.budget.total_chars or "
                    f"lower compression.trigger_chars"
                )

        # Now record the incoming message for future turns. incoming_message_id
        # is None for a normal turn (auto-assign) or a real id for /regen
        # (reuse it) — log_message tells "not given" from "given as None"
        # apart, so which keyword form is used here matters.
        if incoming_message_id is not None:
            logged_id = self.chat_logger.log_message(
                role="user", content=incoming_text, user_id=incoming_user_id, message_id=incoming_message_id
            )
        else:
            logged_id = self.chat_logger.log_message(
                role="user", content=incoming_text, user_id=incoming_user_id
            )

        if archive_incoming:
            await self._remember(logged_id, role="user", content=incoming_text, vector=incoming_vector)

        # Save prompt for debugging
        save_prompt_to_file(messages, self.base_path / "debug" / "last_main_prompt.txt", self.role_names)

        # Generate response
        logger.debug("Calling LLM...")
        response = await self.llm.complete(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            provider_routing=self.provider_routing,
        )

        # Some providers don't cleanly cut the model over from its hidden
        # "reasoning" field to "content" — the whole completion ends up
        # sitting in `reasoning` while `content` comes back null (verified
        # live: same prompt, same model, one run in five landed the full
        # reply in `reasoning` on one provider while four others returned
        # it in `content` normally). This used to be recoverable by looking
        # for a ===LINE=== marker inside `reasoning`; with the response now
        # being bare message text there is nothing to tell a misrouted
        # reply apart from ordinary hidden reasoning, and publishing the
        # latter to the chat is exactly the failure mode that marker
        # guarded against. So: never recover, just record it for diagnosis.
        # .strip() because parse_markers used to do it as a side effect of
        # slicing between markers; providers routinely wrap the completion
        # in leading/trailing newlines and those would otherwise reach the
        # manuscript. Whitespace-only content collapses to None here and
        # falls through to the empty-response branch, which is correct.
        raw_content = response.content.strip() if response.content else None
        if not raw_content and response.reasoning:
            logger.error(
                f"content empty but reasoning is not (provider={response.provider}) — "
                "cannot tell a misrouted reply from ordinary hidden reasoning without a "
                "marker; refusing to publish it. Full reasoning written to debug log"
            )
            append_debug_log(
                {
                    "type": "empty_content_with_reasoning",
                    "provider": response.provider,
                    "finish_reason": response.finish_reason,
                    "reasoning": response.reasoning,
                },
                self.base_path / "debug" / "prompt_log.jsonl",
            )

        if raw_content:
            logger.info(
                f"Generated response ({response.usage.total_tokens if response.usage else 'unknown'} tokens, "
                f"provider={response.provider})"
            )
            logger.debug(f"Response: {raw_content}")

            # The response is the message text itself — no markers to strip.
            line = raw_content

            # Generation cut off mid-message by the token budget. This used
            # to be caught indirectly (a response carrying private reasoning
            # but no message line); asking the provider directly is both
            # simpler and model-agnostic. Fail the turn rather than sending
            # half a sentence — the incoming message stays logged without a
            # reply, so /regen retries it.
            if response.finish_reason == "length":
                logger.error(
                    "Generation hit the token limit mid-message — refusing to send a truncated "
                    "line, turn left unanswered for /regen "
                    f"(provider={response.provider}, usage={response.usage})"
                )
                await self.connector.send(
                    channel_id=channel_id,
                    text="Sorry, I couldn't generate a response.",
                )
                return

            # Log the character's line to the manuscript
            line_id = self.chat_logger.log_message(
                role="assistant",
                content=line,
            )
            await self._remember(line_id, role="assistant", content=line)

            append_debug_log(
                {
                    "type": "generation",
                    "self": self.prompt_builder.self_slug,
                    "other": self.prompt_builder.other_slug,
                    "messages": messages,
                    "raw_response": raw_content,
                    "line": line,
                    "recalled_ids": recalled_ids,
                    "recalled_scores": recalled_scores,
                },
                self.base_path / "debug" / "prompt_log.jsonl",
            )

            # Update state counters with response
            response_chars = len(line)
            self.state_manager.increment_counter("total_chars_since_last_compression", response_chars)

            # COMPRESSION
            if self.compressor.should_compress():
                logger.info("Running compression...")
                result = await self.compressor.compress()
                if not result.get("success"):
                    logger.error(f"Compression failed: {result.get('error')}")
                    # Continue - don't block the user

            # Send response back
            await self.connector.send(
                channel_id=channel_id,
                text=line,
            )
        else:
            logger.warning(
                f"LLM returned empty response (provider={response.provider}, "
                f"finish_reason={response.finish_reason}, usage={response.usage})"
            )
            await self.connector.send(
                channel_id=channel_id,
                text="Sorry, I couldn't generate a response.",
            )

    def _restart_process(self) -> None:
        """Re-exec the whole process (same as `python -m engine.main`) so a
        /switch takes effect. Orchestrator caches a tree of setting-scoped
        objects (PromptBuilder, Compressor, MemoryStore, the LLM provider...)
        in instance attributes at construction time — there is no single
        cached value to update, so rebuilding everything via a fresh process
        is simpler and more reliable than hot-swapping each one in place.
        Sockets Python opens are non-inheritable by default (PEP 446), so the
        Telegram connection closes cleanly across the exec."""
        logger.info("Restarting process for setting switch...")
        os.execv(sys.executable, [sys.executable, "-m", "engine.main"])

    async def _handle_message(self, message: IncomingMessage) -> None:
        """
        Handle incoming message from connector

        Args:
            message: Incoming message to process
        """
        logger.info(
            f"Received message from user {message.user_id} in channel {message.channel_id}"
        )
        logger.debug(f"Message text: {message.text}")

        try:
            # Control commands (e.g. /reset) never reach the LLM or
            # chat_log.jsonl — intercepted before anything else.
            command = parse_command(message.text)
            if command is not None:
                response_text = await self.command_executor.execute(command)
                await self.connector.send(message.channel_id, response_text)
                if self.command_executor.pending_restart:
                    self._restart_process()
                return

            # Check for debug commands
            if self._is_debug_command(message.text):
                await self._handle_debug_command(message.text, message)
                return

            # Update state counters
            message_chars = len(message.text)
            self.state_manager.increment_counter("session_message_count")
            self.state_manager.increment_counter("total_chars_since_last_compression", message_chars)

            # Update user model metadata
            current_user = self.user_model_manager.get_frontmatter()
            if current_user.get("user_id") != message.user_id:
                # New user
                self.user_model_manager.update_frontmatter({
                    "user_id": message.user_id,
                    "first_seen": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                    "interaction_count": 1,
                })
            else:
                # Existing user
                self.user_model_manager.update_frontmatter({
                    "last_seen": datetime.now().isoformat(),
                    "interaction_count": current_user.get("interaction_count", 0) + 1,
                })

            await self._generate_and_send(
                incoming_text=message.text,
                channel_id=message.channel_id,
                incoming_user_id=message.user_id,
            )

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            await self.connector.send(
                channel_id=message.channel_id,
                text=f"Sorry, an error occurred: {str(e)}",
            )

    async def start(self) -> None:
        """Start the orchestrator and connector"""
        logger.info("Starting orchestrator...")
        await self.connector.start()

    async def stop(self) -> None:
        """Stop the orchestrator and connector"""
        logger.info("Stopping orchestrator...")
        await self.connector.stop()
