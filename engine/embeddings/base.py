"""
Base protocol for embedding providers
"""

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Protocol defining the interface for embedding providers"""

    model_name: str

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: Texts to embed. Callers always pass a batch, even for a
                single text, so a turn's query text and incoming message can
                be embedded together in one request.

        Returns:
            One embedding vector per input text, in the same order.
        """
        ...
