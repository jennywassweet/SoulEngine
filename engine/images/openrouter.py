"""
OpenRouter image-generation provider implementation

Note the endpoint: OpenRouter serves image generation from a dedicated
/api/v1/images endpoint, not chat completions, with its own model catalog
that doesn't overlap with /api/v1/models (measured 2026-09-14). Candidate
slugs were found by hand on the OpenRouter site, not derived from the
models list.

Reference images (a character's face, for a consistent likeness) go in the
same request as `input_references` — confirmed against the live API
(2026-09-16). How many a model accepts varies (0 for some, 1, 3, 16) and
is published per model at /api/v1/images/models; it is a property of the
model, so it comes from config as `images.max_references`, not from a
constant here.
"""

import base64
import os
import logging
import httpx

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/images"

# Fallback only — the real value belongs in soul/config.yaml's images block,
# since it's a property of whichever model that setting generates with. 3 is
# what x-ai/grok-imagine-image-2.0 (the default) publishes.
DEFAULT_MAX_REFERENCES = 3


def _data_url(image_bytes: bytes) -> str:
    """Encode raw image bytes as a data: URL for `input_references`.

    Media type comes from the magic bytes rather than a file extension: the
    face artifact next to bible.md may be named .png while actually being a
    JPEG (the image models return JPEG, measured 2026-09-16), and a
    mislabelled content type is rejected by the provider for no visible
    reason.
    """
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif image_bytes.startswith(b"\xff\xd8\xff"):
        media_type = "image/jpeg"
    elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        media_type = "image/webp"
    else:
        raise ValueError("reference image is not a PNG, JPEG or WebP")
    return f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def _build_payload(
    model: str,
    prompt: str,
    reference_images: list[bytes] | None,
    max_references: int,
) -> dict:
    """Assemble the request body. Split out from generate() so the encoding
    rules can be tested without a network call — the HTTP round trip around
    it is a thin wrapper of the kind this project doesn't unit-test."""
    payload: dict = {"model": model, "prompt": prompt}

    if not reference_images:
        # Deliberately absent, not empty: references are billed per image on
        # some providers, and a frame that doesn't use one shouldn't start
        # carrying the field at all.
        return payload

    images = reference_images
    if len(images) > max_references:
        logger.warning(
            f"{len(images)} reference images given but {model} accepts {max_references} — "
            f"sending the first {max_references}"
        )
        images = images[:max_references]

    payload["input_references"] = [
        {"type": "image_url", "image_url": {"url": _data_url(image)}} for image in images
    ]
    return payload


class OpenRouterImageProvider:
    """Image provider backed by OpenRouter's dedicated /images endpoint"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "x-ai/grok-imagine-image-2.0",
        max_references: int = DEFAULT_MAX_REFERENCES,
    ):
        """
        Initialize OpenRouter image provider

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Image model slug, e.g. "x-ai/grok-imagine-image-2.0"
            max_references: How many reference images this model accepts.
                Published per model at /api/v1/images/models — check there
                when changing the model rather than assuming; several models
                in that catalog accept none at all.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be provided or set in environment")

        self.model_name = model
        self.max_references = max_references
        logger.info(
            f"Initialized OpenRouterImageProvider with model: {model} "
            f"(max_references={max_references})"
        )

    async def generate(
        self, prompt: str, reference_images: list[bytes] | None = None
    ) -> bytes:
        """
        Generate one image via OpenRouter's /images endpoint

        Args:
            prompt: Full image-generation prompt
            reference_images: Optional likeness references (e.g. a character's
                face artifact). Anything beyond max_references is dropped with
                a warning rather than failing the request.

        Returns:
            Raw image bytes
        """
        payload = _build_payload(
            self.model_name, prompt, reference_images, self.max_references
        )

        try:
            references = len(payload.get("input_references", []))
            logger.debug(
                f"Calling OpenRouter images API with model {self.model_name} "
                f"({references} reference image(s))"
            )

            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            image = data["data"][0]
            return base64.b64decode(image["b64_json"])

        except Exception as e:
            logger.error(f"Error calling OpenRouter images API: {e}")
            raise
