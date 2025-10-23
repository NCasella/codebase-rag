"""
Base abstractions for LLM providers.

Defines the interface that all LLM providers must implement, enabling
provider-agnostic code generation across different AI services.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """
    Standardized response from any LLM provider.

    Attributes:
        text: The generated text content
        model: Model identifier used for generation
        usage: Token usage statistics (provider-specific format)
        raw_response: Original provider response for debugging
        conversation_id: Unique identifier for continuing multi-turn conversations (provider-specific)
    """
    text: str
    model: str
    usage: dict[str, Any] | None = None
    raw_response: Any = None
    conversation_id: str | None = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Provides a unified interface for text generation across different
    providers (OpenAI, Google, Anthropic, local models, etc.).

    Attributes:
        model: Model identifier string
        api_key: API key for authentication (if required)
    """

    def __init__(self, model: str, api_key: str | None = None):
        """
        Initialize the LLM provider.

        Args:
            model: Model identifier (provider-specific)
            api_key: API authentication key (optional for local models)
        """
        self.model = model
        self.api_key = api_key
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate provider configuration.

        Override this method to add provider-specific validation
        (e.g., check API key exists, validate model name).

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]] | str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        conversation_id: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from the LLM.

        Args:
            messages: Either a simple string prompt or OpenAI-style message array
                     [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (None = provider default)
            conversation_id: Optional conversation ID to continue a previous conversation
            **kwargs: Provider-specific additional parameters

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            Exception: Provider-specific API errors
        """
        pass

    def _format_messages(self, messages: list[dict[str, str]] | str) -> Any:
        """
        Convert between message formats.

        Transforms OpenAI-style messages to provider-specific format.
        Override this method for providers with different message formats.

        Args:
            messages: Input messages (string or message array)

        Returns:
            Provider-specific message format
        """
        if isinstance(messages, str):
            return messages
        return messages

    def continue_conversation(
        self,
        previous_response: LLMResponse,
        new_message: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> LLMResponse:
        """
        Continue a conversation from a previous response.

        This is a convenience method that extracts the conversation_id from
        a previous response and uses it to continue the conversation.

        Args:
            previous_response: The LLMResponse from the previous turn
            new_message: The new user message to send
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (None = provider default)
            **kwargs: Provider-specific additional parameters

        Returns:
            LLMResponse with generated text and updated conversation_id

        Raises:
            NotImplementedError: If the provider doesn't support conversations
            Exception: Provider-specific API errors

        Example:
            >>> response1 = provider.generate("I have 2 dogs")
            >>> response2 = provider.continue_conversation(response1, "How many paws?")
        """
        conversation_id = previous_response.conversation_id

        return self.generate(
            messages=new_message,
            temperature=temperature,
            max_tokens=max_tokens,
            conversation_id=conversation_id,
            **kwargs
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}')"
