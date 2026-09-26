"""
Base protocol for LLM providers
"""

from typing import Protocol
from pydantic import BaseModel


class Usage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ToolCall(BaseModel):
    """Function/tool call from LLM"""
    id: str
    name: str
    arguments: dict


class LLMResponse(BaseModel):
    """Unified response from any LLM provider"""
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None
    finish_reason: str | None = None
    provider: str | None = None
    reasoning: str | None = None


class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers"""

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        reasoning_effort: str | None = None,
        provider_routing: dict | None = None,
    ) -> LLMResponse:
        """
        Generate completion for given messages

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions (function calling)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            reasoning_effort: Optional reasoning effort ("low"/"medium"/"high"),
                ignored by providers/models that don't support it
            provider_routing: Optional OpenRouter `provider` object (order/only/
                ignore/allow_fallbacks/etc.), ignored by providers that don't
                have an equivalent concept

        Returns:
            LLMResponse with content and/or tool calls
        """
        ...
