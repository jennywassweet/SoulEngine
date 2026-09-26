"""
Base protocol for image-generation providers
"""

from typing import Protocol


class ImageProvider(Protocol):
    """Protocol defining the interface for image-generation providers"""

    model_name: str

    async def generate(
        self, prompt: str, reference_images: list[bytes] | None = None
    ) -> bytes:
        """
        Generate one image from a text prompt.

        Args:
            prompt: Full image-generation prompt, already assembled by the
                caller (e.g. PhotoDirector).
            reference_images: Optional reference images (e.g. a character's
                face.png) for providers that support image-to-image or
                likeness conditioning. Present in the signature from v1,
                before any caller passed one, so adding face references
                later needed no breaking change to the Protocol. A provider that doesn't support
                references may raise NotImplementedError if given any.

        Returns:
            Raw image bytes (e.g. PNG).
        """
        ...
