"""
Tests for LLM adapter functionality.

Tests the provider abstraction, factory pattern, and message formatting.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os

from src.llm import (
    BaseLLMProvider,
    LLMResponse,
    Provider,
    GoogleModel,
    OpenAIModel,
    LLMFactory,
    create_llm,
    GoogleProvider,
    OpenAIProvider
)


class TestLLMResponse(unittest.TestCase):
    """Test LLMResponse dataclass."""

    def test_response_creation(self):
        """Test creating an LLMResponse."""
        response = LLMResponse(
            text="Hello world",
            model="test-model",
            usage={"total_tokens": 10},
            raw_response={"foo": "bar"}
        )

        self.assertEqual(response.text, "Hello world")
        self.assertEqual(response.model, "test-model")
        self.assertEqual(response.usage["total_tokens"], 10)
        self.assertEqual(response.raw_response["foo"], "bar")

    def test_response_without_optional_fields(self):
        """Test creating response without usage/raw_response."""
        response = LLMResponse(text="Test", model="model")

        self.assertEqual(response.text, "Test")
        self.assertEqual(response.model, "model")
        self.assertIsNone(response.usage)
        self.assertIsNone(response.raw_response)


class TestProviderEnums(unittest.TestCase):
    """Test provider and model enums."""

    def test_provider_enum(self):
        """Test Provider enum values."""
        self.assertEqual(Provider.OPENAI, "openai")
        self.assertEqual(Provider.GOOGLE, "google")

    def test_google_model_enum(self):
        """Test GoogleModel enum values."""
        self.assertEqual(GoogleModel.GEMINI_2_0_FLASH_EXP, "gemini-2.0-flash-exp")
        self.assertEqual(GoogleModel.GEMINI_1_5_PRO, "gemini-1.5-pro")

    def test_openai_model_enum(self):
        """Test OpenAIModel enum values."""
        self.assertEqual(OpenAIModel.GPT_4O_MINI, "gpt-4o-mini")
        self.assertEqual(OpenAIModel.GPT_4O, "gpt-4o")


class TestMessageFormatting(unittest.TestCase):
    """Test message format conversion between providers."""

    @patch('src.llm.providers.google_provider.genai.Client')
    def test_google_message_formatting_string(self, mock_client):
        """Test Google provider handles string input."""
        provider = GoogleProvider("gemini-2.0-flash-exp", api_key="test-key")

        formatted = provider._format_messages("Hello world")
        self.assertEqual(formatted, "Hello world")

    @patch('src.llm.providers.google_provider.genai.Client')
    def test_google_message_formatting_array(self, mock_client):
        """Test Google provider converts OpenAI-style messages."""
        provider = GoogleProvider("gemini-2.0-flash-exp", api_key="test-key")

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        formatted = provider._format_messages(messages)

        # Should contain system instructions and user message
        self.assertIn("Instructions: You are helpful", formatted)
        self.assertIn("User: Hello", formatted)

    @patch('src.llm.providers.openai_provider.OpenAI')
    def test_openai_message_formatting_string(self, mock_openai):
        """Test OpenAI provider converts string to message array."""
        provider = OpenAIProvider("gpt-4o-mini", api_key="test-key")

        formatted = provider._format_messages("Hello")

        self.assertIsInstance(formatted, list)
        self.assertEqual(len(formatted), 1)
        self.assertEqual(formatted[0]["role"], "user")
        self.assertEqual(formatted[0]["content"], "Hello")

    @patch('src.llm.providers.openai_provider.OpenAI')
    def test_openai_message_formatting_array(self, mock_openai):
        """Test OpenAI provider keeps message array as-is."""
        provider = OpenAIProvider("gpt-4o-mini", api_key="test-key")

        messages = [
            {"role": "system", "content": "System msg"},
            {"role": "user", "content": "User msg"}
        ]

        formatted = provider._format_messages(messages)

        self.assertEqual(formatted, messages)


class TestLLMFactory(unittest.TestCase):
    """Test LLM factory pattern."""

    def setUp(self):
        """Create fresh factory for each test."""
        self.factory = LLMFactory()

    @patch('src.llm.providers.google_provider.genai.Client')
    def test_create_google_provider(self, mock_client):
        """Test creating Google provider through factory."""
        provider = self.factory.create(Provider.GOOGLE, "gemini-2.0-flash-exp", api_key="test")

        self.assertIsInstance(provider, GoogleProvider)
        self.assertEqual(provider.model, "gemini-2.0-flash-exp")

    @patch('src.llm.providers.openai_provider.OpenAI')
    def test_create_openai_provider(self, mock_openai):
        """Test creating OpenAI provider through factory."""
        provider = self.factory.create(Provider.OPENAI, "gpt-4o-mini", api_key="test")

        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.model, "gpt-4o-mini")

    @patch('src.llm.providers.google_provider.genai.Client')
    def test_factory_caching(self, mock_client):
        """Test that factory caches provider instances."""
        provider1 = self.factory.create(Provider.GOOGLE, "gemini-2.0-flash-exp", api_key="test")
        provider2 = self.factory.create(Provider.GOOGLE, "gemini-2.0-flash-exp", api_key="test")

        # Should return same instance from cache
        self.assertIs(provider1, provider2)

    @patch('src.llm.providers.google_provider.genai.Client')
    @patch('src.llm.providers.openai_provider.OpenAI')
    def test_factory_different_providers(self, mock_openai, mock_google):
        """Test factory creates different instances for different providers."""
        google = self.factory.create(Provider.GOOGLE, "gemini-2.0-flash-exp", api_key="test")
        openai = self.factory.create(Provider.OPENAI, "gpt-4o-mini", api_key="test")

        self.assertIsInstance(google, GoogleProvider)
        self.assertIsInstance(openai, OpenAIProvider)
        self.assertIsNot(google, openai)

    def test_factory_invalid_provider(self):
        """Test factory raises error for invalid provider."""
        with self.assertRaises(ValueError):
            self.factory.create("invalid-provider", "model")

    @patch('src.llm.providers.google_provider.genai.Client')
    def test_create_with_string_provider(self, mock_client):
        """Test creating provider with string instead of enum."""
        provider = self.factory.create("google", "gemini-2.0-flash-exp", api_key="test")

        self.assertIsInstance(provider, GoogleProvider)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"})
    @patch('src.llm.providers.google_provider.genai.Client')
    def test_create_from_env_with_gemini(self, mock_client):
        """Test auto-detection prefers Google when GEMINI_API_KEY exists."""
        provider = self.factory.create_from_env()

        self.assertIsInstance(provider, GoogleProvider)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=True)
    @patch('src.llm.providers.openai_provider.OpenAI')
    def test_create_from_env_with_openai(self, mock_openai):
        """Test auto-detection uses OpenAI when only OPENAI_API_KEY exists."""
        # Need to prefer OpenAI in this test
        provider = self.factory.create_from_env(provider_preference=[Provider.OPENAI])

        self.assertIsInstance(provider, OpenAIProvider)

    def test_create_from_env_no_keys(self):
        """Test auto-detection fails when no API keys found."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                self.factory.create_from_env()

            self.assertIn("No API keys found", str(ctx.exception))


class TestProviderValidation(unittest.TestCase):
    """Test provider configuration validation."""

    @patch('src.llm.providers.google_provider.genai.Client')
    def test_google_provider_missing_api_key(self, mock_client):
        """Test Google provider validates API key."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                GoogleProvider("gemini-2.0-flash-exp")

            self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    @patch('src.llm.providers.openai_provider.OpenAI')
    def test_openai_provider_missing_api_key(self, mock_openai):
        """Test OpenAI provider validates API key."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                OpenAIProvider("gpt-4o-mini")

            self.assertIn("OPENAI_API_KEY", str(ctx.exception))


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for creating LLMs."""

    @patch('src.llm.providers.google_provider.genai.Client')
    def test_create_llm_function(self, mock_client):
        """Test create_llm convenience function."""
        llm = create_llm(Provider.GOOGLE, "gemini-2.0-flash-exp", api_key="test")

        self.assertIsInstance(llm, GoogleProvider)


if __name__ == "__main__":
    print("Running LLM adapter tests...")
    print("=" * 60)

    # Run tests
    unittest.main(verbosity=2)
