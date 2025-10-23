"""
Google Gemini LLM provider implementation.

Uses the google-genai library to interact with Google's Gemini models.
"""

import os
from typing import Any

from google import genai

from ..base import BaseLLMProvider, LLMResponse


class GoogleProvider(BaseLLMProvider):
    """
    Google Gemini provider implementation.

    Supports all Gemini models via the google-genai library.

    Example:
        >>> provider = GoogleProvider("gemini-2.0-flash-exp", api_key="...")
        >>> response = provider.generate("How does AI work?")
        >>> print(response.text)
    """

    def __init__(self, model: str, api_key: str | None = None):
        """
        Initialize Google Gemini provider.

        Args:
            model: Gemini model identifier (e.g., "gemini-2.0-flash-exp")
            api_key: Google API key (defaults to GEMINI_API_KEY env var)
        """
        # Get API key from env if not provided
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        super().__init__(model, api_key)

        # Initialize Gemini client
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            # Try without API key (may work with default credentials)
            self.client = genai.Client()

    def _validate_config(self) -> None:
        """
        Validate Google provider configuration.

        Raises:
            ValueError: If API key is missing
        """
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Set it in .env or pass api_key parameter."
            )

    def _format_messages(self, messages: list[dict[str, str]] | str) -> str:
        """
        Convert OpenAI-style messages to Gemini format.

        Gemini's generate_content expects a single string or content parts.
        This method converts OpenAI's message format to a simple string.

        Args:
            messages: String or OpenAI-style message array

        Returns:
            Formatted string for Gemini API
        """
        if isinstance(messages, str):
            return messages

        # Convert message array to formatted string
        # System message -> Instruction prefix
        # User/Assistant -> Conversational format
        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # System messages become instructions
                formatted_parts.append(f"Instructions: {content}\n")
            elif role == "user":
                formatted_parts.append(f"User: {content}\n")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}\n")

        return "\n".join(formatted_parts)

    def generate(
        self,
        messages: list[dict[str, str]] | str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        conversation_id: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using Google Gemini.

        Args:
            messages: Prompt string or OpenAI-style messages
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum output tokens (None for default)
            conversation_id: Not supported yet for Gemini
            **kwargs: Additional Gemini parameters

        Returns:
            LLMResponse with generated text

        Raises:
            NotImplementedError: If conversation_id is provided
            Exception: Gemini API errors
        """
        # Check if conversation continuation is requested
        if conversation_id is not None:
            raise NotImplementedError(
                "Conversation continuation (conversation_id) is not yet supported "
                "for Google Gemini provider. This feature is coming soon."
            )

        # Format messages for Gemini
        prompt = self._format_messages(messages)

        # Build generation config
        config = {
            "temperature": temperature,
        }
        if max_tokens:
            config["max_output_tokens"] = max_tokens

        # Merge with additional kwargs
        config.update(kwargs)

        # Generate content
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )

            # Extract text from response
            # Gemini response has .text attribute
            generated_text = response.text

            # Build LLMResponse
            return LLMResponse(
                text=generated_text,
                model=self.model,
                usage=self._extract_usage(response),
                raw_response=response
            )

        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}") from e

    def _extract_usage(self, response: Any) -> dict[str, Any] | None:
        """
        Extract token usage from Gemini response.

        Args:
            response: Raw Gemini response object

        Returns:
            Usage dictionary or None if unavailable
        """
        try:
            # Gemini API may have usage metadata
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                return {
                    "prompt_tokens": getattr(usage, "prompt_token_count", None),
                    "completion_tokens": getattr(usage, "candidates_token_count", None),
                    "total_tokens": getattr(usage, "total_token_count", None),
                }
        except Exception:
            pass

        return None
