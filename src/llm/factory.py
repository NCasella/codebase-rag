"""
LLM Factory with registry pattern for creating provider instances.

Provides centralized creation and caching of LLM providers.
"""

import os
from typing import Type

from .base import BaseLLMProvider
from .models import Provider, Model, GoogleModel, OpenAIModel
from .providers import GoogleProvider, OpenAIProvider


class LLMFactory:
    """
    Factory for creating and caching LLM provider instances.

    Uses registry pattern to support multiple providers and
    implements singleton caching per (provider, model) combination.

    Example:
        >>> factory = LLMFactory()
        >>> provider = factory.create(Provider.GOOGLE, GoogleModel.GEMINI_2_0_FLASH_EXP)
        >>> response = provider.generate("Hello!")
    """

    # Registry mapping provider enum to provider class
    _PROVIDER_REGISTRY: dict[Provider, Type[BaseLLMProvider]] = {
        Provider.GOOGLE: GoogleProvider,
        Provider.OPENAI: OpenAIProvider,
    }

    def __init__(self):
        """Initialize factory with empty cache."""
        self._cache: dict[tuple[Provider, str], BaseLLMProvider] = {}

    def create(
        self,
        provider: Provider | str,
        model: Model | str,
        api_key: str | None = None,
        force_new: bool = False
    ) -> BaseLLMProvider:
        """
        Create or retrieve cached LLM provider instance.

        Args:
            provider: Provider enum or string ("openai", "google")
            model: Model identifier string
            api_key: Optional API key (defaults to environment variable)
            force_new: If True, create new instance instead of using cache

        Returns:
            BaseLLMProvider instance

        Raises:
            ValueError: If provider is not supported

        Example:
            >>> factory = LLMFactory()
            >>> llm = factory.create("google", "gemini-2.0-flash-exp")
            >>> llm = factory.create(Provider.OPENAI, OpenAIModel.GPT_4O_MINI)
        """
        # Convert string to enum if needed
        if isinstance(provider, str):
            try:
                provider = Provider(provider.lower())
            except ValueError:
                raise ValueError(
                    f"Unsupported provider: {provider}. "
                    f"Supported: {[p.value for p in Provider]}"
                )

        # Check cache
        cache_key = (provider, model)
        if not force_new and cache_key in self._cache:
            return self._cache[cache_key]

        # Get provider class from registry
        provider_class = self._PROVIDER_REGISTRY.get(provider)
        if not provider_class:
            raise ValueError(
                f"Provider {provider} not registered. "
                f"Available: {list(self._PROVIDER_REGISTRY.keys())}"
            )

        # Create instance
        instance = provider_class(model=model, api_key=api_key)

        # Cache instance
        self._cache[cache_key] = instance

        return instance

    def create_from_env(
        self,
        model: str | None = None,
        provider_preference: list[Provider] | None = None
    ) -> BaseLLMProvider:
        """
        Create provider by auto-detecting from environment variables.

        If model is specified, infers provider from model name.
        Otherwise checks for API keys in environment and selects provider accordingly.

        Args:
            model: Model to use (None = use default for detected provider)
                  If provided, provider is inferred from model name
            provider_preference: List of providers in preference order
                                (default: [Google, OpenAI])
                                Only used if model is not specified

        Returns:
            BaseLLMProvider instance

        Raises:
            ValueError: If no API keys found in environment or model provider cannot be inferred

        Example:
            >>> factory = LLMFactory()
            >>> # Auto-detect provider from env vars
            >>> llm = factory.create_from_env()
            >>> # Use specific model (provider inferred)
            >>> llm = factory.create_from_env(model="gpt-4o")
        """
        # If model specified, infer provider from model name
        if model:
            provider = self._infer_provider_from_model(model)
            if provider:
                # Verify API key exists
                env_var = self._get_env_var_for_provider(provider)
                if not os.getenv(env_var):
                    raise ValueError(
                        f"Model '{model}' requires {env_var} to be set in environment"
                    )
                return self.create(provider, model)
            else:
                raise ValueError(
                    f"Cannot infer provider from model '{model}'. "
                    "Specify provider explicitly or use standard model names."
                )

        # No model specified - use provider preference
        if provider_preference is None:
            provider_preference = [Provider.GOOGLE, Provider.OPENAI]

        # Try each provider in preference order
        for provider in provider_preference:
            env_var = self._get_env_var_for_provider(provider)
            if os.getenv(env_var):
                # Found API key for this provider
                default_model = self._get_default_model(provider)
                return self.create(provider, default_model)

        # No API keys found
        available_vars = [self._get_env_var_for_provider(p) for p in provider_preference]
        raise ValueError(
            f"No API keys found in environment. Set one of: {available_vars}"
        )

    @staticmethod
    def _get_env_var_for_provider(provider: Provider) -> str:
        """Get environment variable name for provider's API key."""
        env_vars = {
            Provider.GOOGLE: "GEMINI_API_KEY",
            Provider.OPENAI: "OPENAI_API_KEY",
        }
        return env_vars[provider]

    @staticmethod
    def _get_default_model(provider: Provider) -> str:
        """Get default model for provider."""
        defaults = {
            Provider.GOOGLE: GoogleModel.GEMINI_2_5_FLASH_LITE,
            Provider.OPENAI: OpenAIModel.GPT_5_MINI,
        }
        return defaults[provider]

    @staticmethod
    def _infer_provider_from_model(model: str) -> Provider | None:
        """
        Infer provider from model name by checking enum membership.

        Args:
            model: Model identifier string

        Returns:
            Provider enum or None if cannot be inferred
        """
        # Check if model is a GoogleModel
        if model in [m.value for m in GoogleModel]:
            return Provider.GOOGLE

        # Check if model is an OpenAIModel
        if model in [m.value for m in OpenAIModel]:
            return Provider.OPENAI

        # Fallback: try prefix matching for unknown models
        model_lower = model.lower()
        if model_lower.startswith("gemini"):
            return Provider.GOOGLE
        if model_lower.startswith("gpt"):
            return Provider.OPENAI

        # Cannot infer
        return None

    @classmethod
    def register_provider(
        cls,
        provider: Provider,
        provider_class: Type[BaseLLMProvider]
    ) -> None:
        """
        Register a new provider class (for extensibility).

        Args:
            provider: Provider enum
            provider_class: Provider implementation class

        Example:
            >>> class MyCustomProvider(BaseLLMProvider):
            ...     # implementation
            >>> LLMFactory.register_provider(Provider.CUSTOM, MyCustomProvider)
        """
        cls._PROVIDER_REGISTRY[provider] = provider_class

    def clear_cache(self) -> None:
        """Clear all cached provider instances."""
        self._cache.clear()

    def get_cached_providers(self) -> list[tuple[Provider, str]]:
        """
        Get list of cached (provider, model) combinations.

        Returns:
            List of (provider, model) tuples
        """
        return list(self._cache.keys())


# Global singleton factory instance for convenience
_global_factory = LLMFactory()


def create_llm(
    provider: Provider | str,
    model: str,
    api_key: str | None = None
) -> BaseLLMProvider:
    """
    Convenience function to create LLM using global factory.

    Args:
        provider: Provider enum or string
        model: Model identifier
        api_key: Optional API key

    Returns:
        BaseLLMProvider instance

    Example:
        >>> from src.llm import create_llm, Provider, GoogleModel
        >>> llm = create_llm(Provider.GOOGLE, GoogleModel.GEMINI_2_0_FLASH_EXP)
    """
    return _global_factory.create(provider, model, api_key)


def create_llm_from_env(model: str | None = None) -> BaseLLMProvider:
    """
    Convenience function to create LLM from environment using global factory.

    Args:
        model: Optional model override

    Returns:
        BaseLLMProvider instance

    Example:
        >>> from src.llm import create_llm_from_env
        >>> llm = create_llm_from_env()  # Auto-detects provider from env
    """
    return _global_factory.create_from_env(model)
