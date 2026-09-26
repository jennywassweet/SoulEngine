"""
Probe reference-image (likeness) conditioning for the setting's image model
(first run 2026-09-16).

Answers, by measurement rather than by guess, the four questions that decide
whether a consistent face is workable at all with the setting's image model:

  1. does the face carry over from the reference into a new frame;
  2. does the reference leak its *pose / background / framing* into the
     frame (the main risk: the model redraws the reference instead of
     shooting the described scene);
  3. does prompt text override the reference's mutable attributes (hair
     colour is the marker - the whole point of the "identity anchor"
     decision is that everything but bone structure is set by the scene);
  4. does moderation behave differently once a photorealistic face is
     attached (zero refusals on text-only prompts does NOT transfer).

Like scripts/probe_image_models.py this talks to the API directly rather
than through engine/images/ - the probe must stay independent of the
production code it is supposed to inform (and at the time the provider could not
send references yet anyway).

Two stages, run separately because a human looks at the output in between:

    set -a; source .env; set +a
    uv run python scripts/probe_face_reference.py anchor
    # look at debug/face_probe/anchor_*.png, pick one
    uv run python scripts/probe_face_reference.py frames --anchor debug/face_probe/anchor_2.png

Cost, at x-ai/grok-imagine-image-2.0 rates (2026-09-16): $0.06 per image
plus $0.01 per attached reference, so the default run is ~$0.18 for three
anchor candidates and ~$0.33 for the five frames.
"""

import argparse
import asyncio
import base64
import os
import sys
import time
from pathlib import Path

import httpx

API_URL = "https://openrouter.ai/api/v1/images"
DEFAULT_MODEL = "x-ai/grok-imagine-image-2.0"

# Physical traits only, the way config/photo_prompt.md tells the describer to
# read a bible. These are the traits the probe was originally run against;
# --traits replaces them for any other character.
DEFAULT_TRAITS = (
    "a 24-year-old Hungarian woman, slim, angular, androgynous build, "
    "short dark brown pixie haircut, brown eyes"
)

# The anchor is an *identity* reference, not a portrait in character: plain
# light, plain background, nothing about the frame that we would mind seeing
# copied into every generated scene. Everything mutable (makeup, hair colour,
# lenses, expression, wardrobe) is deliberately absent here so that the scene
# prompt is the only thing that can set it - that is exactly what case
# `ref_hair_override` below tests.
ANCHOR_PROMPT = (
    "Neutral identity reference photograph of {traits}. Plain even lighting, "
    "plain light grey background, head and shoulders, facing the camera, "
    "neutral relaxed expression, no makeup, natural undyed hair, sharp "
    "focus, no stylization, no filter - like a casting or passport reference "
    "photo."
)

# Genre wording deliberately copied in spirit from config/photo_prompt.md so
# the probe measures the real thing, not a prettier cousin of it.
SELFIE_PROMPT = (
    "Vertical phone selfie taken at arm's length in a cramped circus trailer "
    "at night: a young androgynous woman in her twenties sitting on the edge "
    "of a bunk in mismatched tights and an oversized cardigan, harsh "
    "overhead bulb washing out one side of her face, tilted horizon, visible "
    "grain, slight motion blur, cluttered shelf of props out of focus behind "
    "her. Candid and unglamorous, like a real photo sent in a text message, "
    "never a professional portrait. The phone itself is not visible in the "
    "frame."
)

# Deliberately far from the anchor in every non-identity dimension - outdoors,
# night, full body, in motion, seen from behind-ish - so that any leakage of
# the anchor's framing or background is obvious at a glance.
SCENE_PROMPT = (
    "Vertical phone snapshot taken by someone standing a few metres away, "
    "outdoors at night behind a fairground: a young androgynous woman in her "
    "twenties in a stage costume and a man's canvas jacket, full body, "
    "caught mid-stride across wet gravel between parked trailers, head "
    "turned back over her shoulder towards the camera, lit only by a sodium "
    "floodlight and the glow of a ferris wheel far behind. Crooked framing, "
    "heavy grain, motion blur, puddles, cables underfoot. Candid, unposed, "
    "like a real photo from a phone, not a professional shot."
)

# Same camera as SELFIE_PROMPT, but every mutable attribute is set loudly
# against the anchor (which has natural dark hair, no makeup, neutral eyes).
# If the reference wins here, the "identity anchor" decision does not hold
# and either the anchor or the model has to change.
HAIR_OVERRIDE_PROMPT = (
    "Vertical phone selfie taken at arm's length in a cramped circus trailer "
    "at night: a young androgynous woman in her twenties with BRIGHT GREEN "
    "dyed hair and vivid blue contact lenses, wearing heavy white and red "
    "clown greasepaint across her whole face, sitting on the edge of a bunk. "
    "Harsh overhead bulb, tilted horizon, visible grain, slight motion blur. "
    "Candid and unglamorous, like a real photo sent in a text message. The "
    "phone itself is not visible in the frame."
)

# Close to the register of the setting this was developed against - the
# cluster of signals that has tripped text-model moderation repeatedly here.
# Nothing explicit; the question is whether attaching a real-looking face
# changes the provider's answer.
MODERATION_PROMPT = (
    "Vertical phone photo, early morning light through a trailer window: a "
    "young woman in her twenties half-asleep in tangled bedsheets, bare "
    "shoulder and messy hair above the blanket line, smeared remains of "
    "yesterday's stage makeup, phone held at arm's length, slight motion "
    "blur, grainy, nothing explicit visible."
)

# label -> (prompt, attach_reference)
FRAME_CASES = {
    "baseline_selfie": (SELFIE_PROMPT, False),
    "ref_selfie": (SELFIE_PROMPT, True),
    "ref_scene": (SCENE_PROMPT, True),
    "ref_hair_override": (HAIR_OVERRIDE_PROMPT, True),
    "ref_moderation": (MODERATION_PROMPT, True),
}

WHAT_TO_LOOK_AT = """
What to judge by eye (this is the output of 6.0, not the console table):

  baseline_selfie vs ref_selfie
      Is it recognisably the same person as the anchor? That is question 1.
      Compare the two against each other too - baseline shows what the model
      invents on its own, which is the thing the reference is meant to replace.

  ref_scene
      Night, outdoors, full body, mid-stride. If the anchor's grey background,
      head-and-shoulders framing or flat lighting show up here, the model is
      redrawing the reference instead of shooting the scene - question 2, and
      the one that would sink the whole approach.

  ref_hair_override
      Green hair, blue lenses, full greasepaint were all asked for in text and
      none of them are in the anchor. Whatever wins here decides question 3.
      If the anchor's natural dark hair survives, the identity-anchor design
      does not hold as written.

  ref_moderation
      Compare against the text-only moderation batch in
      scripts/probe_image_models.py: a refusal here but not there means
      likeness conditioning has its own filter - worth
      knowing before /selfie is opened in a live setting.
"""


def data_url(path: Path) -> str:
    """Encode an image file as a data: URL. Media type comes from the magic
    bytes, not the extension - a mislabelled .png would otherwise be sent
    with a content type the provider rejects for no visible reason."""
    raw = path.read_bytes()
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        media_type = "image/jpeg"
    elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        media_type = "image/webp"
    else:
        raise ValueError(f"{path}: not a PNG, JPEG or WebP image")
    return f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}"


async def call_one(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    prompt: str,
    reference: str | None = None,
) -> dict:
    """One /images call. Reference images go in `input_references` as
    {"type": "image_url", "image_url": {"url": ...}} - confirmed against the
    live API 2026-09-16."""
    payload: dict = {"model": model, "prompt": prompt}
    if reference:
        payload["input_references"] = [
            {"type": "image_url", "image_url": {"url": reference}}
        ]

    started = time.monotonic()
    try:
        response = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        elapsed = time.monotonic() - started
        data = response.json()
        if response.status_code != 200 or "error" in data:
            message = data.get("error", {}).get("message", response.text[:300])
            return {"ok": False, "elapsed": elapsed, "error": message}
        image = data["data"][0]
        return {
            "ok": True,
            "elapsed": elapsed,
            "media_type": image.get("media_type", "image/png"),
            "b64": image["b64_json"],
            "cost": data.get("usage", {}).get("cost"),
        }
    except Exception as e:
        return {"ok": False, "elapsed": time.monotonic() - started, "error": str(e)}


def save_image(out_dir: Path, label: str, result: dict) -> Path:
    ext = result["media_type"].split("/")[-1]
    path = out_dir / f"{label}.{ext}"
    path.write_bytes(base64.b64decode(result["b64"]))
    return path


def report(label: str, result: dict, out_dir: Path) -> float:
    if result["ok"]:
        path = save_image(out_dir, label, result)
        cost = result["cost"] or 0.0
        print(f"  OK    {label:20s} {result['elapsed']:6.1f}s  ${cost:.4f}  -> {path}")
        return cost
    print(f"  FAIL  {label:20s} {result['elapsed']:6.1f}s  {result['error']}")
    return 0.0


async def run_anchor(args: argparse.Namespace, api_key: str, out_dir: Path) -> None:
    prompt = ANCHOR_PROMPT.format(traits=args.traits)
    print(f"\n=== anchor candidates ({args.n}) ===\n{prompt}\n")
    async with httpx.AsyncClient(timeout=300.0) as client:
        results = await asyncio.gather(
            *(call_one(client, api_key, args.model, prompt) for _ in range(args.n))
        )
    total = sum(report(f"anchor_{i + 1}", r, out_dir) for i, r in enumerate(results))
    print(f"\n  total ${total:.4f}")
    print("\nPick one by eye, then run:")
    print(f"  uv run python scripts/probe_face_reference.py frames --anchor {out_dir}/anchor_1.png")


async def run_frames(args: argparse.Namespace, api_key: str, out_dir: Path) -> None:
    anchor_path = Path(args.anchor)
    if not anchor_path.exists():
        print(f"anchor not found: {anchor_path} — run the `anchor` stage first.", file=sys.stderr)
        sys.exit(1)
    reference = data_url(anchor_path)

    print(f"\n=== frames (anchor: {anchor_path}, {anchor_path.stat().st_size / 1024:.0f} KB) ===\n")
    async with httpx.AsyncClient(timeout=300.0) as client:
        results = await asyncio.gather(
            *(
                call_one(client, api_key, args.model, prompt, reference if use_ref else None)
                for prompt, use_ref in FRAME_CASES.values()
            )
        )
    total = sum(report(label, r, out_dir) for label, r in zip(FRAME_CASES, results))
    print(f"\n  total ${total:.4f}")
    print(WHAT_TO_LOOK_AT)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["anchor", "frames"])
    parser.add_argument("--anchor", default="debug/face_probe/anchor_1.png",
                        help="reference image for the `frames` stage")
    parser.add_argument("--traits", default=DEFAULT_TRAITS,
                        help="physical traits for the `anchor` stage, as a bible states them")
    parser.add_argument("--n", type=int, default=3, help="anchor candidates to generate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="debug/face_probe")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set — source .env first.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "anchor":
        await run_anchor(args, api_key, out_dir)
    else:
        await run_frames(args, api_key, out_dir)


if __name__ == "__main__":
    asyncio.run(main())
