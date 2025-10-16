"""
LLM Adapter - Unified interface for multiple LLM providers.

Provides provider-agnostic text generation across OpenAI, Google Gemini,
and potentially other AI services.

Example:
    >>> from src.llm import create_llm, Provider, GoogleModel
    >>> llm = create_llm(Provider.GOOGLE, GoogleModel.GEMINI_2_0_FLASH_EXP)
    >>> response = llm.generate("Explain recursion")
    >>> print(response.text)

    >>> # Or auto-detect from environment
    >>> from src.llm import create_llm_from_env
    >>> llm = create_llm_from_env()
    >>> response = llm.generate("How does AI work?")
"""

from .base import BaseLLMProvider, LLMResponse
from .models import Provider, GoogleModel, OpenAIModel, get_model_metadata, supports_temperature
from .factory import LLMFactory, create_llm, create_llm_from_env
from .providers import GoogleProvider, OpenAIProvider

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "LLMResponse",

    # Enums
    "Provider",
    "GoogleModel",
    "OpenAIModel",

    # Functions
    "get_model_metadata",
    "supports_temperature",
    "create_llm",
    "create_llm_from_env",

    # Classes
    "LLMFactory",
    "GoogleProvider",
    "OpenAIProvider",
]
