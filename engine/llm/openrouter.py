"""
OpenRouter provider implementation
"""

import os
import json
import logging
import httpx

from engine.llm.base import LLMProvider, LLMResponse, ToolCall, Usage

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider:
    """OpenRouter LLM provider (OpenAI-compatible chat completions API)"""

    def __init__(self, api_key: str | None = None, model: str = "anthropic/claude-sonnet-4.5"):
        """
        Initialize OpenRouter provider

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model slug to use, e.g. "anthropic/claude-sonnet-4.5"
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be provided or set in environment")

        self.model = model
        logger.info(f"Initialized OpenRouterProvider with model: {model}")

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
        Generate completion using OpenRouter's chat completions API

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            reasoning_effort: Optional reasoning effort ("low"/"medium"/"high"),
                passed through to models that support OpenRouter's `reasoning` field
            provider_routing: Optional OpenRouter `provider` object (e.g.
                {"order": ["deepinfra"], "allow_fallbacks": False}) — passed
                through verbatim, see https://openrouter.ai/docs/features/provider-routing

        Returns:
            LLMResponse with completion
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if provider_routing:
            payload["provider"] = provider_routing

        try:
            logger.debug(f"Calling OpenRouter API with {len(messages)} messages")

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

            # OpenRouter reports upstream provider failures (content
            # moderation, provider outage, quota) as HTTP 200 with an
            # "error" body and no "choices" — raise_for_status() sees
            # nothing wrong. Without this the caller gets a bare
            # KeyError: 'choices' and the real reason is lost.
            if "error" in data:
                error = data["error"]
                raise RuntimeError(
                    f"OpenRouter returned an error: {error.get('message', error)} "
                    f"(code {error.get('code', 'unknown')})"
                )

            choice = data["choices"][0]
            message = choice["message"]

            tool_calls = None
            if message.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in message["tool_calls"]
                ]

            usage = None
            if data.get("usage"):
                usage = Usage(
                    prompt_tokens=data["usage"]["prompt_tokens"],
                    completion_tokens=data["usage"]["completion_tokens"],
                    total_tokens=data["usage"]["total_tokens"],
                )

            content = message.get("content")
            reasoning = message.get("reasoning")
            finish_reason = choice.get("finish_reason")

            logger.debug(f"Received response: {usage.total_tokens if usage else 'unknown'} tokens")

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                finish_reason=finish_reason,
                provider=data.get("provider"),
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            raise
