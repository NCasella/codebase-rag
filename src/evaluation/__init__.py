"""
RAGAS evaluation module for Codebase RAG.

Provides comprehensive RAG system evaluation using RAGAS metrics including:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Features:
- Isolated evaluator LLM (separate from generation)
- Metric dependency validation
- Version tracking for reproducibility
- Error handling and logging
- Ground truth construction utilities
"""

from .ragas_evaluator import RAGASEvaluator, EvaluationConfig, ValidationReport
from .test_dataset_generator import TestDatasetGenerator

__all__ = [
    "RAGASEvaluator",
    "EvaluationConfig",
    "ValidationReport",
    "TestDatasetGenerator"
]
