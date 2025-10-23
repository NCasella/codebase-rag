"""
Tests for RAGAS evaluation module.

Tests validation, dataset preparation, metric requirements, and error handling.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path

from src.evaluation import RAGASEvaluator, EvaluationConfig, ValidationReport, TestDatasetGenerator
from src.evaluation.ragas_evaluator import METRIC_REQUIREMENTS


class TestEvaluationConfig(unittest.TestCase):
    """Test EvaluationConfig loading and validation."""

    def test_config_from_json(self):
        """Test loading config from JSON file."""
        # Create temporary config file
        config_data = {
            "name": "test_config",
            "description": "Test configuration",
            "evaluator_llm": {
                "provider": "google",
                "model": "gemini-1.5-flash",
                "temperature": 0.0
            },
            "evaluator_embeddings": {
                "model_name": "all-MiniLM-L6-v2"
            },
            "metrics": ["faithfulness", "answer_relevancy"],
            "batch_size": 5
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = EvaluationConfig.from_json(temp_path)

            self.assertEqual(config.name, "test_config")
            self.assertEqual(config.evaluator_llm["provider"], "google")
            self.assertEqual(len(config.metrics), 2)
            self.assertEqual(config.batch_size, 5)

        finally:
            Path(temp_path).unlink()

    def test_config_missing_file(self):
        """Test error when config file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            EvaluationConfig.from_json("nonexistent_config.json")

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = EvaluationConfig(
            name="test",
            description="test desc",
            evaluator_llm={"provider": "google"},
            evaluator_embeddings={"model_name": "test-model"},
            metrics=["faithfulness"]
        )

        config_dict = config.to_dict()

        self.assertEqual(config_dict["name"], "test")
        self.assertIn("evaluator_llm", config_dict)
        self.assertIn("metrics", config_dict)


class TestValidationReport(unittest.TestCase):
    """Test ValidationReport functionality."""

    def test_valid_report(self):
        """Test valid validation report."""
        report = ValidationReport(valid=True)

        self.assertTrue(report.valid)
        self.assertEqual(len(report.warnings), 0)
        self.assertEqual(len(report.errors), 0)

    def test_invalid_report_with_errors(self):
        """Test invalid report with errors."""
        report = ValidationReport(
            valid=False,
            errors=["Missing required field"],
            warnings=["Optional field missing"]
        )

        self.assertFalse(report.valid)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(len(report.warnings), 1)

    def test_report_string_representation(self):
        """Test string representation of report."""
        report = ValidationReport(
            valid=True,
            warnings=["Test warning"]
        )

        report_str = str(report)

        self.assertIn("VALIDATION REPORT", report_str)
        self.assertIn("VALID", report_str)
        self.assertIn("Test warning", report_str)


class TestMetricRequirements(unittest.TestCase):
    """Test metric dependency matrix."""

    def test_faithfulness_requirements(self):
        """Test faithfulness metric requirements."""
        reqs = METRIC_REQUIREMENTS["faithfulness"]

        self.assertIn("user_input", reqs["required"])
        self.assertIn("response", reqs["required"])
        self.assertIn("retrieved_contexts", reqs["required"])
        self.assertFalse(reqs["needs_reference"])

    def test_context_precision_requirements(self):
        """Test context_precision metric requirements."""
        reqs = METRIC_REQUIREMENTS["context_precision"]

        self.assertIn("reference", reqs["required"])
        self.assertTrue(reqs["needs_reference"])

    def test_answer_relevancy_requirements(self):
        """Test answer_relevancy metric requirements."""
        reqs = METRIC_REQUIREMENTS["answer_relevancy"]

        self.assertIn("user_input", reqs["required"])
        self.assertIn("response", reqs["required"])
        self.assertFalse(reqs["needs_reference"])


class TestRAGASEvaluator(unittest.TestCase):
    """Test RAGASEvaluator functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.config = EvaluationConfig(
            name="test_eval",
            description="Test evaluation",
            evaluator_llm={
                "provider": "google",
                "model": "gemini-1.5-flash",
                "temperature": 0.0,
                "api_key_env": "GEMINI_API_KEY"
            },
            evaluator_embeddings={
                "model_name": "all-MiniLM-L6-v2"
            },
            metrics=["faithfulness", "answer_relevancy"],
            verbose=False
        )

    @patch('src.evaluation.ragas_evaluator.ChatGoogleGenerativeAI')
    @patch('src.evaluation.ragas_evaluator.OpenAIEmbeddings')
    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'})
    def test_evaluator_initialization(self, mock_embeddings, mock_llm):
        """Test evaluator initialization."""
        evaluator = RAGASEvaluator(self.config)

        self.assertEqual(evaluator.config.name, "test_eval")
        self.assertIsNotNone(evaluator.evaluator_llm)
        self.assertIsNotNone(evaluator.evaluator_embeddings)

    def test_validate_dataset_valid(self):
        """Test dataset validation with valid data."""
        with patch('src.evaluation.ragas_evaluator.ChatGoogleGenerativeAI'), \
             patch('src.evaluation.ragas_evaluator.OpenAIEmbeddings'), \
             patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):

            evaluator = RAGASEvaluator(self.config)

            dataset = [
                {
                    "user_input": "Test question?",
                    "response": "Test answer",
                    "retrieved_contexts": ["Context 1"]
                }
            ]

            report = evaluator.validate_dataset(dataset)

            self.assertTrue(report.valid)

    def test_validate_dataset_missing_fields(self):
        """Test dataset validation with missing fields."""
        with patch('src.evaluation.ragas_evaluator.ChatGoogleGenerativeAI'), \
             patch('src.evaluation.ragas_evaluator.OpenAIEmbeddings'), \
             patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):

            evaluator = RAGASEvaluator(self.config)

            # Dataset missing 'response' field
            dataset = [
                {
                    "user_input": "Test question?",
                    "retrieved_contexts": ["Context 1"]
                }
            ]

            report = evaluator.validate_dataset(dataset)

            self.assertFalse(report.valid)
            self.assertGreater(len(report.errors), 0)

    def test_validate_empty_dataset(self):
        """Test validation of empty dataset."""
        with patch('src.evaluation.ragas_evaluator.ChatGoogleGenerativeAI'), \
             patch('src.evaluation.ragas_evaluator.OpenAIEmbeddings'), \
             patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):

            evaluator = RAGASEvaluator(self.config)

            report = evaluator.validate_dataset([])

            self.assertFalse(report.valid)
            self.assertIn("empty", report.errors[0].lower())

    def test_prepare_evaluation_dataset(self):
        """Test preparing evaluation dataset."""
        with patch('src.evaluation.ragas_evaluator.ChatGoogleGenerativeAI'), \
             patch('src.evaluation.ragas_evaluator.OpenAIEmbeddings'), \
             patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):

            evaluator = RAGASEvaluator(self.config)

            queries = ["Q1", "Q2"]
            responses = ["A1", "A2"]
            contexts = [["C1"], ["C2"]]

            dataset, report = evaluator.prepare_evaluation_dataset(
                queries, responses, contexts
            )

            self.assertTrue(report.valid)
            self.assertEqual(len(dataset), 2)

    def test_prepare_dataset_length_mismatch(self):
        """Test error on length mismatch in prepare_evaluation_dataset."""
        with patch('src.evaluation.ragas_evaluator.ChatGoogleGenerativeAI'), \
             patch('src.evaluation.ragas_evaluator.OpenAIEmbeddings'), \
             patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):

            evaluator = RAGASEvaluator(self.config)

            queries = ["Q1", "Q2"]
            responses = ["A1"]  # Length mismatch
            contexts = [["C1"], ["C2"]]

            with self.assertRaises(ValueError):
                evaluator.prepare_evaluation_dataset(queries, responses, contexts)


class TestTestDatasetGenerator(unittest.TestCase):
    """Test TestDatasetGenerator functionality."""

    def setUp(self):
        """Set up test generator."""
        self.generator = TestDatasetGenerator()

    def test_load_from_json(self):
        """Test loading queries from JSON."""
        test_data = {
            "samples": [
                {
                    "question": "Test Q1?",
                    "reference": "Test A1",
                    "metadata": {"category": "test"}
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            queries = self.generator.load_from_json(temp_path)

            self.assertEqual(len(queries), 1)
            self.assertEqual(queries[0].question, "Test Q1?")
            self.assertEqual(queries[0].reference, "Test A1")

        finally:
            Path(temp_path).unlink()

    def test_load_from_json_missing_file(self):
        """Test error when JSON file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.generator.load_from_json("nonexistent.json")

    def test_save_to_json(self):
        """Test saving queries to JSON."""
        from src.evaluation.test_dataset_generator import TestQuery

        queries = [
            TestQuery("Q1?", "A1"),
            TestQuery("Q2?", None)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_queries.json"

            self.generator.save_to_json(queries, str(output_path))

            self.assertTrue(output_path.exists())

            # Verify saved data
            with open(output_path) as f:
                data = json.load(f)

            self.assertEqual(len(data["samples"]), 2)
            self.assertEqual(data["metadata"]["num_samples"], 2)
            self.assertEqual(data["metadata"]["with_references"], 1)

    def test_generate_from_codebase(self):
        """Test auto-generating queries from codebase."""
        mock_collection = Mock()

        queries = self.generator.generate_from_codebase(
            collection=mock_collection,
            num_samples=5
        )

        self.assertEqual(len(queries), 5)
        for query in queries:
            self.assertTrue(query.question)
            self.assertIsNone(query.reference)  # Auto-generated don't have refs

    def test_validate_dataset(self):
        """Test dataset validation."""
        from src.evaluation.test_dataset_generator import TestQuery

        queries = [
            TestQuery("Q1?", "A1"),
            TestQuery("Q2?", None)
        ]

        report = self.generator.validate_dataset(
            queries,
            metrics=["faithfulness", "context_precision"]
        )

        self.assertIn("stats", report)
        self.assertEqual(report["stats"]["total_queries"], 2)
        self.assertEqual(report["stats"]["with_references"], 1)


class TestVersionMetadata(unittest.TestCase):
    """Test version tracking functionality."""

    @patch('src.evaluation.ragas_evaluator.ChatGoogleGenerativeAI')
    @patch('src.evaluation.ragas_evaluator.OpenAIEmbeddings')
    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'})
    def test_get_version_metadata(self, mock_embeddings, mock_llm):
        """Test version metadata generation."""
        config = EvaluationConfig(
            name="test",
            description="test",
            evaluator_llm={
                "provider": "google",
                "model": "gemini-1.5-flash",
                "api_key_env": "GEMINI_API_KEY"
            },
            evaluator_embeddings={"model_name": "test-model"},
            metrics=["faithfulness"]
        )

        evaluator = RAGASEvaluator(config)
        metadata = evaluator._get_version_metadata()

        self.assertIn("timestamp", metadata)
        self.assertIn("python_version", metadata)
        self.assertIn("ragas_version", metadata)
        self.assertIn("evaluator_model", metadata)
        self.assertIn("embedding_model", metadata)


if __name__ == "__main__":
    print("Running RAGAS evaluator tests...")
    print("=" * 60)

    unittest.main(verbosity=2)
