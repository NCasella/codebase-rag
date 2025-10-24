"""
Model enums and metadata for supported LLM providers.

Defines available models for each provider with their specifications.
"""

from enum import StrEnum


class Provider(StrEnum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GOOGLE = "google"


class Model(StrEnum):
    """Supported LLM models."""
    pass

class GoogleModel(Model):
    """
    Google Gemini models.

    See: https://ai.google.dev/gemini-api/docs/models
    """
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"


class OpenAIModel(Model):
    """
    OpenAI GPT models.

    See: https://platform.openai.com/docs/models
    """
    GPT_5_MINI = "gpt-5-mini"
    GPT_4_1_MINI = "gpt-4.1-mini"

# Model metadata for advanced features (context windows, pricing, etc.)
MODEL_METADATA = {
    # Google Gemini
    GoogleModel.GEMINI_2_5_FLASH_LITE: {
        "context_window": 1_048_576,
        "output_limit": 65_536,
        "tier": "production",
        "supports_temperature": True
    },
    GoogleModel.GEMINI_2_0_FLASH_LITE: {
        "context_window": 1_048_576,
        "output_limit": 65_536,
        "tier": "production",
        "supports_temperature": True
    },

    # OpenAI GPT
    OpenAIModel.GPT_5_MINI: {
        "context_window": 400_000,
        "output_limit": 128_000,
        "tier": "production",
        "supports_temperature": False  # Only supports default 1.0
    },
    OpenAIModel.GPT_4_1_MINI: {
        "context_window": 1_047_576,
        "output_limit": 32_768,
        "tier": "production",
        "supports_temperature": True
    }
}


def get_model_metadata(model: str) -> dict:
    """
    Get metadata for a model.

    Args:
        model: Model identifier string

    Returns:
        Dictionary with context_window, output_limit, tier, supports_temperature
        Empty dict if model not found
    """
    return MODEL_METADATA.get(model, {})


def supports_temperature(model: str) -> bool:
    """
    Check if a model supports custom temperature values.

    Args:
        model: Model identifier string

    Returns:
        True if model supports temperature, False otherwise.
        Defaults to True if model not found in metadata (permissive).
    """
    metadata = get_model_metadata(model)
    # Default to True for unknown models (permissive)
    return metadata.get("supports_temperature", True)
