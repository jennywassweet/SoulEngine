"""
Generate candidate face artifacts for a character.

`face.png` is an authoring artifact of the same class as `bible.md`: made
once, chosen by eye, committed to git, never regenerated behind your back.
That is why this is a command-line tool and not a chat command — a command
in the chat would invite re-rolling the character's face, which is the one
thing the design forbids. It is also why nothing here writes the artifact
itself unless you pass --accept.

What it does, in two calls:

  1. the setting's own describer model reads the character's bible and
     writes an English identity-anchor prompt. This step exists because a
     bible is prose in whatever language it was authored in, while image
     models want a short English physical description — and because "which traits are identity" is a judgement
     the describer already makes for every frame.
  2. the setting's own image model renders N candidates from that prompt.

The anchor is deliberately *neutral*: plain light, plain background, no
makeup, no styling. Everything mutable about a character's look — makeup,
hair colour, lenses, expression — is set per-frame by the describer, and
was measured (2026-09-16) to override the reference reliably. An anchor
that carries styling of its own would leak that styling into every frame
instead.

Usage:
    set -a; source .env; set +a
    uv run python scripts/make_face.py                     # active setting's `self`
    uv run python scripts/make_face.py --n 4
    uv run python scripts/make_face.py --setting other_setting --slug name
    uv run python scripts/make_face.py --extra "softer jawline, early 30s"
    uv run python scripts/make_face.py --accept 2          # adopt candidate 2, no generation

Candidates land in debug/faces/<slug>/ (gitignored). --accept copies one
next to bible.md, which is where PhotoDirector looks for it.
"""

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config_utils import read_config, resolve_images_config
from engine.images.openrouter import OpenRouterImageProvider
from engine.llm.openrouter import OpenRouterProvider
from engine.photo_director import FACE_FILENAMES
from engine.prompt_builder import load_markdown_file

# Mechanics, not genre: what makes a usable identity anchor is a property of
# how reference conditioning works, not of any one character's aesthetic.
# Per-setting steering goes through --extra rather than a forked file - this
# is a tool run twice per character, not a prompt that wants iterating on.
ANCHOR_INSTRUCTION = """You write prompts for an image-generation model.

Read the character bible below and write ONE English prompt for a neutral
identity reference photograph of this character - the kind of plain photo
used to establish what a person looks like, not a portrait with mood.

Rules:
- Physical identity only: apparent age stated plainly and neutrally, build,
  face shape, hair colour and cut in their natural/default state, eye
  colour, skin, distinguishing features.
- No makeup, no stage look, no costume, no props, no expression beyond
  neutral, no styling of any kind. If the bible says the character is
  usually seen made up, dyed or in character - ignore that; this is the
  face underneath.
- Plain even lighting, plain light grey background, head and shoulders,
  facing the camera, sharp focus, no filter, no stylization.
- Photographic, not illustrated.

Respond with the prompt text alone - no preamble, no quotes, no JSON."""


def _media_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "webp"
    return "bin"


def _character_dir(base: Path, setting: str, slug: str) -> Path:
    return base / "soul" / "settings" / setting / "characters" / slug


def _accept(base: Path, setting: str, slug: str, candidates_dir: Path, number: int) -> None:
    """Copy one candidate next to bible.md, where PhotoDirector looks for it."""
    matches = sorted(candidates_dir.glob(f"cand_{number}.*"))
    if not matches:
        print(f"No candidate {number} in {candidates_dir} — generate some first.", file=sys.stderr)
        sys.exit(1)

    source = matches[0]
    character_dir = _character_dir(base, setting, slug)
    if not character_dir.is_dir():
        print(f"No such character directory: {character_dir}", file=sys.stderr)
        sys.exit(1)

    existing = [character_dir / name for name in FACE_FILENAMES if (character_dir / name).exists()]
    if existing:
        print("This character already has a face artifact:")
        for path in existing:
            print(f"  {path}")
        print("Delete or rename it first — replacing a face silently is exactly what this")
        print("tool is built not to do.")
        sys.exit(1)

    destination = character_dir / f"face.{source.suffix.lstrip('.')}"
    shutil.copyfile(source, destination)
    print(f"Adopted {source} -> {destination}")
    print("It is NOT gitignored: commit it alongside bible.md.")


async def _write_anchor_prompt(describer_config: dict, bible: str, extra: str) -> str:
    llm_kwargs = {}
    if describer_config.get("model"):
        llm_kwargs["model"] = describer_config["model"]
    llm = OpenRouterProvider(**llm_kwargs)

    instruction = ANCHOR_INSTRUCTION
    if extra:
        instruction += f"\n\nAdditional direction from the author (overrides the bible where they conflict):\n{extra}"

    response = await llm.complete(
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"CHARACTER BIBLE:\n{bible}"},
        ],
        temperature=describer_config.get("temperature", 0.8),
        max_tokens=describer_config.get("max_tokens", 800),
        reasoning_effort=describer_config.get("reasoning_effort"),
    )
    text = (response.content or "").strip()
    if not text:
        raise RuntimeError(
            "The describer returned nothing. If this model routes its output through a "
            "reasoning field or tripped a content filter, see the notes in "
            "soul/settings/*/images.yaml for models already known to work."
        )
    return text


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", help="defaults to soul/config.yaml's active setting")
    parser.add_argument("--slug", help="defaults to that setting's `self` character")
    parser.add_argument("--n", type=int, default=3, help="candidates to generate")
    parser.add_argument("--extra", default="", help="author's steering, layered over the bible")
    parser.add_argument("--accept", type=int, metavar="N",
                        help="adopt candidate N as this character's face and exit")
    parser.add_argument("--out", default="debug/faces")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    config = read_config(base)
    setting = args.setting or config.get("setting", "default")
    slug = args.slug or config.get("characters", {}).get("self", "self")
    candidates_dir = Path(args.out) / slug

    if args.accept is not None:
        _accept(base, setting, slug, candidates_dir, args.accept)
        return

    images_config = resolve_images_config(base, setting, config.get("images", {}))
    bible_path = _character_dir(base, setting, slug) / "bible.md"
    bible = load_markdown_file(bible_path)
    if not bible:
        print(f"No bible at {bible_path} — wrong --setting/--slug?", file=sys.stderr)
        sys.exit(1)

    print(f"setting: {setting}   character: {slug}")
    print(f"bible:   {bible_path} ({len(bible)} chars)")

    prompt = await _write_anchor_prompt(images_config.get("llm", {}), bible, args.extra)
    print(f"\n=== anchor prompt ===\n{prompt}\n")

    image_kwargs = {}
    if images_config.get("model"):
        image_kwargs["model"] = images_config["model"]
    image_provider = OpenRouterImageProvider(**image_kwargs)

    candidates_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== generating {args.n} candidates with {image_provider.model_name} ===")
    results = await asyncio.gather(
        *(image_provider.generate(prompt) for _ in range(args.n)),
        return_exceptions=True,
    )

    for index, result in enumerate(results, start=1):
        if isinstance(result, Exception):
            print(f"  FAIL  cand_{index}: {result}")
            continue
        path = candidates_dir / f"cand_{index}.{_media_extension(result)}"
        path.write_bytes(result)
        print(f"  OK    {path} ({len(result) / 1024:.0f} KB)")

    print("\nLook at them, then adopt one:")
    print(f"  uv run python scripts/make_face.py --setting {setting} --slug {slug} --accept 1")


if __name__ == "__main__":
    asyncio.run(main())
