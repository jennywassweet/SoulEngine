"""
Embedding providers abstraction and implementations
"""

from engine.embeddings.base import EmbeddingProvider
from engine.embeddings.openrouter import OpenRouterEmbeddingProvider

__all__ = ["EmbeddingProvider", "OpenRouterEmbeddingProvider"]
