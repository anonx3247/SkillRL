"""DeepSeek API client wrapper with retry logic."""

import asyncio
import os
from typing import Any

from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError, InternalServerError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class DeepSeekClient:
    """Wrapper for DeepSeek API using OpenAI-compatible interface."""

    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat"):
        """Initialize DeepSeek client.

        Args:
            api_key: API key (defaults to DEEPSEEK_API_KEY env var)
            model: Model to use (default: deepseek-chat for function calling)

        Note: Use "deepseek-chat" for function calling, NOT "deepseek-reasoner"
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")

        self.model = model
        self.client = AsyncOpenAI(
            base_url="https://api.deepseek.com",
            api_key=self.api_key,
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, InternalServerError)
        ),
    )
    async def _chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> Any:
        """Internal method with retry logic."""
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=temperature,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> Any:
        """Make chat completion request with timeout and retry.

        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool definitions (OpenAI function calling format)
            temperature: Sampling temperature (default: 0.7)

        Returns:
            ChatCompletion response object

        Raises:
            asyncio.TimeoutError: If request exceeds 120 seconds
            APIError: If request fails after retries
        """
        try:
            return await asyncio.wait_for(
                self._chat_with_retry(messages, tools, temperature),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError("DeepSeek API request timed out after 120 seconds")
