"""
PhotoDirector - assembles context, calls the frame describer, and (if a
frame is available) calls the image provider and records the result.

Deliberately does not use PromptBuilder.build_prompt() - that glues the
epistolary system frame and the generation trigger, the fiction circuit
this engine stays outside of. Everything here reads bible / relationship /
continuity / manuscript read-only; nothing is logged back to chat_logger,
memory, or continuity.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from engine.chat_logger import ChatLogger
from engine.config_utils import RUNTIME_DIR_NAME
from engine.debug_logger import save_prompt_to_file
from engine.images.base import ImageProvider
from engine.llm.base import LLMProvider
from engine.photo_store import PhotoStore
from engine.prompt_builder import load_markdown_file

logger = logging.getLogger(__name__)

CAMERA_DESCRIPTIONS = {
    "scene": "an unseen observer in the room; her face and the world around her are both in frame",
    "shot": "her own phone, her point of view; her face is NOT in frame",
    "selfie": "her phone, pointed at herself; her face MUST be in frame",
}

# Whether each camera position sends the character's face artifact as a
# likeness reference. "required" means the mode doesn't come up at all
# without one - a selfie with a different face every time is exactly what
# the face artifact exists to fix; "optional" means the frame is better with
# one but works without, so a setting that has no face artifact yet keeps
# the mode it already had. Decided 2026-09-16.
FACE_POLICY = {
    "shot": "never",      # her face is not in frame by definition
    "scene": "optional",
    "selfie": "required",
}

# Accepted spellings of the face artifact, in preference order. Three rather
# than one because the image models return JPEG (measured) and there is
# nothing in this project to convert with - Pillow is deliberately not a
# dependency. The media type is read from the bytes either way.
FACE_FILENAMES = ("face.png", "face.jpg", "face.jpeg")

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class PhotoModeUnavailable(Exception):
    """Raised when a mode isn't enabled for this setting - a config/permission
    rejection, not an in-fiction refusal. No LLM/image call is made and
    nothing is recorded to PhotoStore for this: no scene was ever evaluated."""


class PhotoDirector:
    """Assembles context, calls the describer, and produces one PhotoStore record per request"""

    def __init__(
        self,
        llm: LLMProvider,
        image_provider: ImageProvider,
        photo_store: PhotoStore,
        chat_logger: ChatLogger,
        setting_path: Path | str,
        self_slug: str,
        role_names: dict[str, str],
        config: dict[str, Any],
        prompt_file: Path | str,
        describer_model: str,
        template_vars: dict[str, str] | None = None,
        debug_prompt_path: Path | str | None = None,
    ):
        """
        Args:
            llm: Text model that decides availability and writes the image prompt
            image_provider: Generates the actual frame
            photo_store: Where every request (including refusals) is recorded
            chat_logger: Read-only source for the manuscript tail
            setting_path: soul/settings/<setting> - source of self-bible,
                relationship, and continuity (read-only)
            self_slug: Character slug whose bible is read. There is
                deliberately no other-bible: the frame is requested by a
                reader of the correspondence, not by the other character,
                so the camera belongs to this character's world alone
            role_names: role -> display name, for formatting the manuscript tail
            config: Resolved `images:` config (enabled, modes, max_chars, ...)
            prompt_file: Resolved describer prompt template - config/photo_prompt.md
                or a per-setting override, already resolved by the caller
            describer_model: Name of the llm model, for the photos.jsonl
                record (LLMProvider doesn't expose this itself, unlike
                ImageProvider/EmbeddingProvider)
            template_vars: Extra {{key}} substitutions for the prompt template
            debug_prompt_path: If given, the last describer prompt is saved
                here whole (e.g. base_path/debug/last_photo_prompt.txt) -
                built by the caller from base_path, never from CWD
        """
        self.llm = llm
        self.image_provider = image_provider
        self.photo_store = photo_store
        self.chat_logger = chat_logger
        self.setting_path = Path(setting_path)
        self.runtime_path = self.setting_path / RUNTIME_DIR_NAME
        self.self_slug = self_slug
        self.role_names = role_names
        self.config = config
        self.prompt_file = Path(prompt_file)
        self.describer_model = describer_model
        self.template_vars = template_vars or {}
        self.debug_prompt_path = Path(debug_prompt_path) if debug_prompt_path else None

        self.enabled = config.get("enabled", False)
        self.modes = config.get("modes", [])
        self.max_chars = config.get("max_chars", 12000)

        describer_llm_config = config.get("llm", {})
        self.temperature = describer_llm_config.get("temperature", 0.8)
        self.max_tokens = describer_llm_config.get("max_tokens", 800)
        # Verified live (2026-09-14) against google/gemini-3.8-flash: without
        # this, the model spends most of max_tokens on its own hidden
        # reasoning before ever emitting the JSON, and the response comes
        # back truncated (finish_reason="length") more often than not - the
        # same behavior the main text llm.reasoning_effort setting already
        # works around for this model, see soul/config.yaml.
        self.reasoning_effort = describer_llm_config.get("reasoning_effort")

        # Fail at startup, not at the first command: a mode that silently
        # never works is the same trap as a bible that loads as an empty
        # string when the slug is wrong (see CLAUDE.md).
        missing = [
            mode
            for mode in self.modes
            if FACE_POLICY.get(mode, "never") == "required" and self._face_path() is None
        ]
        if missing:
            raise ValueError(
                f"images.yaml enables {missing} for this setting, but no face artifact "
                f"({' / '.join(FACE_FILENAMES)}) exists in "
                f"{self.setting_path / 'characters' / self.self_slug}. "
                "Those modes need a consistent likeness to be worth anything - generate "
                "one with scripts/make_face.py and put it next to bible.md, or drop the "
                "mode from images.yaml."
            )

    def _face_path(self) -> Path | None:
        """The character's face artifact, if this setting has one. Resolved per
        call rather than cached at startup: the file is tiny, and replacing a
        face without restarting the bot is exactly what scripts/make_face.py
        exists for."""
        directory = self.setting_path / "characters" / self.self_slug
        for name in FACE_FILENAMES:
            candidate = directory / name
            if candidate.exists():
                return candidate
        return None

    def _resolve_reference(self, mode: str) -> tuple[list[bytes] | None, str | None]:
        """Reference images for one frame, plus the file name for the record.

        Raises:
            PhotoModeUnavailable: the mode requires a face and it isn't there.
                Normally impossible (__init__ validates it), but the file can
                be deleted while the bot runs.
        """
        policy = FACE_POLICY.get(mode, "never")
        if policy == "never":
            return None, None

        face_path = self._face_path()
        if face_path is None:
            if policy == "required":
                raise PhotoModeUnavailable(
                    f"mode '{mode}' needs a face artifact for {self.self_slug} and none is present"
                )
            return None, None

        return [face_path.read_bytes()], face_path.name

    def _load_prompt_template(self) -> str:
        template = load_markdown_file(self.prompt_file)
        for key, value in self.template_vars.items():
            template = template.replace("{{" + key + "}}", value)
        return template

    def _format_manuscript_tail(self) -> str:
        messages = self.chat_logger.read_last_n_chars(self.max_chars)
        lines = [
            f"{self.role_names.get(entry.get('role', 'user'), entry.get('role', ''))}: {entry.get('content', '')}"
            for entry in messages
        ]
        return "\n".join(lines)

    def _build_context(self, mode: str, hint: str, reference_attached: bool = False) -> str:
        self_bible = load_markdown_file(self.setting_path / "characters" / self.self_slug / "bible.md")
        relationship = load_markdown_file(self.setting_path / "relationship.md")
        continuity = load_markdown_file(self.runtime_path / "continuity.md")
        manuscript = self._format_manuscript_tail()
        camera = CAMERA_DESCRIPTIONS.get(mode, mode)

        parts = [
            f"BIBLE:\n{self_bible}" if self_bible else "",
            f"RELATIONSHIP:\n{relationship}" if relationship else "",
            f"CONTINUITY:\n{continuity}" if continuity else "",
            f"CORRESPONDENCE (tail):\n{manuscript}" if manuscript else "",
            f"CAMERA: {mode} - {camera}",
            # A bare fact about this call, like CAMERA. What it means for the
            # prompt the describer writes lives in the prompt template, so a
            # setting can fork that reasoning without forking this code.
            f"REFERENCE: {'attached' if reference_attached else 'none attached'}",
            f"READER'S HINT: {hint}" if hint else "READER'S HINT: (none - the present moment)",
        ]
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any] | None:
        """Parse the describer's JSON reply. Tolerates a markdown code fence
        around it (the prompt asks for none, but real output doesn't always
        comply); anything else unparseable returns None rather than raising."""
        cleaned = _CODE_FENCE_RE.sub("", raw.strip()).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "available" not in data:
            return None
        return data

    async def generate(self, mode: str, hint: str = "") -> dict[str, Any]:
        """
        Run one photo request end to end: decide, describe, and (if
        available) generate and store the frame. Always returns a
        PhotoStore record for an evaluated scene - a malformed describer
        reply is recorded as no-frame-available rather than raised.

        Args:
            mode: "shot" / "scene" / "selfie"
            hint: Reader's pointer at a moment ("that night on the roof"), or ""

        Returns:
            The PhotoStore record for this request (see PhotoStore.record).

        Raises:
            PhotoModeUnavailable: images are disabled globally, or `mode`
                isn't in this setting's configured modes. Nothing is
                recorded - this is a permission rejection, not a scene
                that said no.
        """
        if not self.enabled:
            raise PhotoModeUnavailable("images are disabled (images.enabled=false)")
        if mode not in self.modes:
            raise PhotoModeUnavailable(f"mode '{mode}' is not enabled for this setting (modes={self.modes})")

        reference_images, reference_name = self._resolve_reference(mode)

        prompt_template = self._load_prompt_template()
        context = self._build_context(mode, hint, reference_attached=bool(reference_images))
        messages = [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": context},
        ]

        if self.debug_prompt_path:
            save_prompt_to_file(messages, self.debug_prompt_path, self.role_names)

        response = await self.llm.complete(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        parsed = self._parse_response(response.content or "")

        if parsed is None:
            logger.error(f"Photo describer returned unparseable output: {(response.content or '')[:500]}")
            return self.photo_store.record(
                mode=mode,
                hint=hint,
                available=False,
                prompt="",
                note="no frame right now",
                model=self.describer_model,
            )

        available = bool(parsed.get("available"))
        prompt = str(parsed.get("prompt") or "")
        note = str(parsed.get("note") or "")

        if not available:
            return self.photo_store.record(
                mode=mode,
                hint=hint,
                available=False,
                prompt="",
                note=note,
                model=self.describer_model,
            )

        image_bytes = await self.image_provider.generate(
            prompt, reference_images=reference_images
        )

        return self.photo_store.record(
            mode=mode,
            hint=hint,
            available=True,
            prompt=prompt,
            note=note,
            model=self.describer_model,
            image_model=self.image_provider.model_name,
            # ImageProvider.generate() returns raw bytes only (engine/images/base.py) -
            # OpenRouter's usage.cost isn't surfaced through the Protocol yet.
            cost=None,
            reference=reference_name,
            image_bytes=image_bytes,
        )

    async def again(self, modifier: str = "") -> dict[str, Any] | None:
        """
        Regenerate the last delivered frame, optionally revising its prompt.
        Unlike generate(), this skips the describer call entirely - it edits
        and resubmits the last stored prompt directly to the image
        provider. This is the only form of continuity between frames - a
        reader revising a shot, still from outside the fiction.

        Args:
            modifier: Text appended to the last prompt before resubmitting
                it (e.g. "more light"), or "" to just re-roll the same prompt.

        Returns:
            None if there's no prior delivered frame to revise. Otherwise
            the new PhotoStore record.

        Raises:
            PhotoModeUnavailable: images are disabled globally, or the last
                frame's mode is no longer enabled for this setting.
        """
        if not self.enabled:
            raise PhotoModeUnavailable("images are disabled (images.enabled=false)")

        last = self.photo_store.last_available()
        if last is None:
            return None

        mode = last["mode"]
        if mode not in self.modes:
            raise PhotoModeUnavailable(f"mode '{mode}' is not enabled for this setting (modes={self.modes})")

        # Resolved fresh rather than copied from the old record: the face
        # artifact may have been added or replaced since that frame was made,
        # and /again is the natural way to re-roll a frame against a new one.
        reference_images, reference_name = self._resolve_reference(mode)

        prompt = f"{last['prompt']} {modifier}".strip() if modifier else last["prompt"]
        image_bytes = await self.image_provider.generate(
            prompt, reference_images=reference_images
        )

        return self.photo_store.record(
            mode=mode,
            hint=last["hint"],
            available=True,
            prompt=prompt,
            note=last["note"],
            model=last["model"],
            image_model=self.image_provider.model_name,
            cost=None,
            reference=reference_name,
            image_bytes=image_bytes,
        )
