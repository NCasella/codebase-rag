"""
OpenAI GPT provider implementation.

Uses the openai library to interact with OpenAI's GPT models.
"""

import os
from typing import Any

from openai import OpenAI

from ..base import BaseLLMProvider, LLMResponse
from ..models import supports_temperature


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI GPT provider implementation.

    Supports all OpenAI models (GPT-4, GPT-3.5, etc.) via the openai library.

    Example:
        >>> provider = OpenAIProvider("gpt-4o-mini", api_key="...")
        >>> response = provider.generate("Explain recursion")
        >>> print(response.text)
    """

    def __init__(self, model: str, api_key: str | None = None):
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model identifier (e.g., "gpt-4o-mini")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        # Get API key from env if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        super().__init__(model, api_key)

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def _validate_config(self) -> None:
        """
        Validate OpenAI provider configuration.

        Raises:
            ValueError: If API key is missing
        """
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Set it in .env or pass api_key parameter."
            )

    def _format_messages(self, messages: list[dict[str, str]] | str) -> list[dict[str, str]]:
        """
        Convert to OpenAI message format.

        OpenAI expects message arrays. If a string is provided,
        convert it to a single user message.

        Args:
            messages: String or message array

        Returns:
            OpenAI-compatible message array
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return messages

    def generate(
        self,
        messages: list[dict[str, str]] | str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        conversation_id: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using OpenAI Responses API.

        Uses the OpenAI Responses API which provides native support for
        multi-turn conversations via previous_response_id.

        Args:
            messages: Prompt string or OpenAI-style messages
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens (None for default)
            conversation_id: Previous response ID to continue a conversation
            **kwargs: Additional OpenAI parameters (top_p, store, include, etc.)

        Returns:
            LLMResponse with generated text and conversation_id for continuation

        Raises:
            Exception: OpenAI API errors

        Note:
            This implementation uses the Responses API (client.responses.create)
            instead of Chat Completions API for better conversation support.
        """
        # Format messages for OpenAI
        formatted_messages = self._format_messages(messages)

        # Build API parameters for Responses API
        api_params = {
            "model": self.model,
            "input": formatted_messages,
        }

        # Only add temperature if model supports it
        if supports_temperature(self.model):
            api_params["temperature"] = temperature

        if max_tokens:
            api_params["max_tokens"] = max_tokens

        # Add previous_response_id for conversation continuity
        if conversation_id:
            api_params["previous_response_id"] = conversation_id

        # Merge with additional kwargs
        api_params.update(kwargs)

        # Generate response using Responses API
        try:
            response = self.client.responses.create(**api_params)

            # Extract text from response using helper method
            generated_text = response.output_text

            # Extract response ID for conversation continuity
            response_id = getattr(response, "id", None)

            # Build LLMResponse
            return LLMResponse(
                text=generated_text,
                model=self.model,
                usage=self._extract_usage(response),
                raw_response=response,
                conversation_id=response_id
            )

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}") from e

    def _extract_usage(self, response: Any) -> dict[str, Any] | None:
        """
        Extract token usage from OpenAI response.

        Args:
            response: Raw OpenAI response object

        Returns:
            Usage dictionary or None if unavailable
        """
        try:
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                return {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
        except Exception:
            pass

        return None
