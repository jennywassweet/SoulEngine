"""
Mistral AI provider implementation
"""

import os
import logging
from mistralai.client import Mistral

from engine.llm.base import LLMProvider, LLMResponse, ToolCall, Usage

logger = logging.getLogger(__name__)


class MistralProvider:
    """Mistral AI LLM provider"""

    def __init__(self, api_key: str | None = None, model: str = "mistral-medium-latest"):
        """
        Initialize Mistral provider

        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            model: Model name to use
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY must be provided or set in environment")

        self.model = model
        self.client = Mistral(api_key=self.api_key)
        logger.info(f"Initialized MistralProvider with model: {model}")

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
        Generate completion using Mistral API (reasoning_effort and
        provider_routing are ignored; Mistral has no equivalent of either)

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with completion
        """
        try:
            logger.debug(f"Calling Mistral API with {len(messages)} messages")

            # Call Mistral API
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
            )

            # Extract message from response
            choice = response.choices[0]
            message = choice.message

            # Parse tool calls if present
            tool_calls = None
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in message.tool_calls
                ]

            # Parse usage
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            content = message.content if hasattr(message, 'content') else None
            finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else None

            logger.debug(f"Received response: {usage.total_tokens if usage else 'unknown'} tokens")

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                finish_reason=finish_reason,
            )

        except Exception as e:
            logger.error(f"Error calling Mistral API: {e}")
            raise
