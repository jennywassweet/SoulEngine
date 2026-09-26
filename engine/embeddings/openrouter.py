"""
OpenRouter embedding provider implementation
"""

import os
import logging
import httpx

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/embeddings"


class OpenRouterEmbeddingProvider:
    """Embedding provider backed by OpenRouter's OpenAI-compatible /embeddings endpoint"""

    def __init__(self, api_key: str | None = None, model: str = "openai/text-embedding-3-small"):
        """
        Initialize OpenRouter embedding provider

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Embedding model slug, e.g. "openai/text-embedding-3-small"
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be provided or set in environment")

        self.model_name = model
        logger.info(f"Initialized OpenRouterEmbeddingProvider with model: {model}")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts via OpenRouter's OpenAI-compatible endpoint

        Args:
            texts: Texts to embed

        Returns:
            One embedding vector per input text, in the same order
        """
        payload = {
            "model": self.model_name,
            "input": texts,
        }

        try:
            logger.debug(f"Calling OpenRouter embeddings API with {len(texts)} texts")

            async with httpx.AsyncClient(timeout=120.0) as client:
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

            # OpenAI-compatible shape: data["data"] is a list of
            # {"index": i, "embedding": [...]}. Sort by index rather than
            # trusting response order, per the API's own documented contract.
            ordered = sorted(data["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in ordered]

        except Exception as e:
            logger.error(f"Error calling OpenRouter embeddings API: {e}")
            raise
