"""
RAGAS Evaluator for Codebase RAG.

Provides comprehensive evaluation functionality with:
- Metric dependency validation
- Isolated evaluator LLM
- Version tracking for reproducibility
- Error handling and logging
"""

import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import pandas as pd

# RAGAS imports
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    LLMContextRecall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain imports for LLM wrapping
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from src.llm import Provider


# Metric dependency matrix
METRIC_REQUIREMENTS = {
    "faithfulness": {
        "required": ["user_input", "response", "retrieved_contexts"],
        "optional": [],
        "needs_reference": False
    },
    "answer_relevancy": {
        "required": ["user_input", "response"],
        "optional": ["retrieved_contexts"],
        "needs_reference": False
    },
    "context_precision": {
        "required": ["user_input", "retrieved_contexts", "reference"],
        "optional": [],
        "needs_reference": True
    },
    "context_recall": {
        "required": ["user_input", "retrieved_contexts", "reference"],
        "optional": [],
        "needs_reference": True
    },
    "llm_context_recall": {
        "required": ["user_input", "retrieved_contexts", "reference"],
        "optional": [],
        "needs_reference": True
    }
}

# Available RAGAS metrics mapping
AVAILABLE_METRICS = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "llm_context_recall": LLMContextRecall
}


@dataclass
class ValidationReport:
    """Report of dataset validation against metric requirements."""
    valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    missing_fields: Dict[str, List[str]] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable validation report."""
        lines = [f"\n{'='*60}"]
        lines.append("VALIDATION REPORT")
        lines.append('='*60)
        lines.append(f"Status: {'✅ VALID' if self.valid else '❌ INVALID'}")

        if self.errors:
            lines.append(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"   • {error}")

        if self.warnings:
            lines.append(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"   • {warning}")

        if self.missing_fields:
            lines.append(f"\n📋 Missing Fields by Metric:")
            for metric, fields in self.missing_fields.items():
                lines.append(f"   • {metric}: {', '.join(fields)}")

        lines.append('='*60)
        return '\n'.join(lines)


@dataclass
class EvaluationConfig:
    """Configuration for RAGAS evaluation."""
    name: str
    description: str
    evaluator_llm: Dict[str, Any]
    evaluator_embeddings: Dict[str, Any]
    metrics: List[str]
    batch_size: int = 10
    require_reference: bool = False
    warn_on_missing_fields: bool = True
    stop_on_error: bool = False
    verbose: bool = False

    @classmethod
    def from_json(cls, json_path: str) -> 'EvaluationConfig':
        """Load evaluation configuration from JSON file."""
        path = Path(json_path)

        if not path.exists():
            raise FileNotFoundError(f"Evaluation config not found: {json_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'evaluator_llm': self.evaluator_llm,
            'evaluator_embeddings': self.evaluator_embeddings,
            'metrics': self.metrics,
            'batch_size': self.batch_size,
            'require_reference': self.require_reference,
            'warn_on_missing_fields': self.warn_on_missing_fields,
            'stop_on_error': self.stop_on_error,
            'verbose': self.verbose
        }


class RAGASEvaluator:
    """
    RAGAS evaluator with validation, isolation, and version tracking.

    Example:
        >>> config = EvaluationConfig.from_json("configs/evaluation.json")
        >>> evaluator = RAGASEvaluator(config)
        >>> dataset = [...] # List of evaluation samples
        >>> results = evaluator.evaluate(dataset, verbose=True)
        >>> evaluator.export_results(results, "results/eval.csv")
    """

    def __init__(self, eval_config: EvaluationConfig):
        """
        Initialize RAGAS evaluator with isolated LLM.

        Args:
            eval_config: Evaluation configuration with isolated evaluator LLM
        """
        self.config = eval_config
        self.evaluator_llm = self._create_evaluator_llm()
        self.evaluator_embeddings = self._create_evaluator_embeddings()
        self.sample_errors: List[Dict[str, Any]] = []

        if self.config.verbose:
            print(f"\n🔧 Initialized RAGASEvaluator: {self.config.name}")
            print(f"   Evaluator LLM: {self.config.evaluator_llm['provider']}/{self.config.evaluator_llm['model']}")
            print(f"   Embeddings: {self.config.evaluator_embeddings['model_name']}")
            print(f"   Metrics: {', '.join(self.config.metrics)}")

    def _create_evaluator_llm(self) -> LangchainLLMWrapper:
        """Create isolated evaluator LLM (separate from generation LLM)."""
        llm_config = self.config.evaluator_llm
        provider = llm_config['provider'].lower()
        model = llm_config['model']

        # Get API key from environment
        api_key_env = llm_config.get('api_key_env')
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(
                    f"API key not found in environment: {api_key_env}. "
                    f"Please set it in your .env file."
                )
        else:
            api_key = None

        # Create LangChain LLM based on provider
        if provider == "openai":
            llm = ChatOpenAI(
                model=model,
                temperature=llm_config.get('temperature', 0.0),
                api_key=api_key
            )
        elif provider == "google":
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=llm_config.get('temperature', 0.0),
                google_api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Wrap for RAGAS compatibility
        return LangchainLLMWrapper(llm)

    def _create_evaluator_embeddings(self) -> LangchainEmbeddingsWrapper:
        """Create embeddings for evaluator."""
        emb_config = self.config.evaluator_embeddings

        # For now, use OpenAI embeddings (RAGAS default)
        # TODO: Support SentenceTransformer embeddings
        embeddings = OpenAIEmbeddings(
            model=emb_config.get('model_name', 'text-embedding-ada-002')
        )

        return LangchainEmbeddingsWrapper(embeddings)

    def validate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Validate dataset against metric requirements.

        Args:
            dataset: List of evaluation samples
            metrics: Metrics to validate for (defaults to config.metrics)

        Returns:
            ValidationReport with validation status and warnings
        """
        if metrics is None:
            metrics = self.config.metrics

        report = ValidationReport(valid=True)

        # Check if dataset is empty
        if not dataset:
            report.valid = False
            report.errors.append("Dataset is empty")
            return report

        # Validate each metric's requirements
        for metric_name in metrics:
            if metric_name not in METRIC_REQUIREMENTS:
                report.warnings.append(
                    f"Unknown metric '{metric_name}' - skipping validation"
                )
                continue

            requirements = METRIC_REQUIREMENTS[metric_name]
            required_fields = requirements['required']
            needs_reference = requirements['needs_reference']

            # Check if any sample has all required fields
            samples_with_fields = 0
            missing_fields_count = {field: 0 for field in required_fields}

            for i, sample in enumerate(dataset):
                sample_valid = True
                for field in required_fields:
                    if field not in sample or sample[field] is None:
                        missing_fields_count[field] += 1
                        sample_valid = False

                if sample_valid:
                    samples_with_fields += 1

            # Report missing fields
            for field, count in missing_fields_count.items():
                if count > 0:
                    pct = (count / len(dataset)) * 100
                    msg = f"Field '{field}' missing in {count}/{len(dataset)} samples ({pct:.1f}%) for metric '{metric_name}'"

                    if count == len(dataset):
                        # All samples missing this field - error if required
                        report.errors.append(msg)
                        report.valid = False
                        if metric_name not in report.missing_fields:
                            report.missing_fields[metric_name] = []
                        report.missing_fields[metric_name].append(field)
                    elif self.config.warn_on_missing_fields:
                        # Some samples missing - warning
                        report.warnings.append(msg)

            # Special warning for reference requirement
            if needs_reference and 'reference' in missing_fields_count and missing_fields_count['reference'] > 0:
                report.warnings.append(
                    f"⚠️  Metric '{metric_name}' requires 'reference' (ground truth) for accurate evaluation. "
                    f"Consider using TestDatasetGenerator.build_reference_answers() to generate references."
                )

        return report

    def prepare_evaluation_dataset(
        self,
        queries: List[str],
        responses: List[str],
        contexts: List[List[str]],
        references: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[EvaluationDataset, ValidationReport]:
        """
        Prepare dataset for RAGAS evaluation with validation.

        Args:
            queries: List of user queries
            responses: List of generated responses
            contexts: List of retrieved contexts (each item is a list of context strings)
            references: Optional ground truth answers
            metadata: Optional metadata for each sample

        Returns:
            (EvaluationDataset, ValidationReport)
        """
        # Validate input lengths
        n_samples = len(queries)
        if len(responses) != n_samples:
            raise ValueError(f"Length mismatch: {len(queries)} queries, {len(responses)} responses")
        if len(contexts) != n_samples:
            raise ValueError(f"Length mismatch: {len(queries)} queries, {len(contexts)} contexts")
        if references and len(references) != n_samples:
            raise ValueError(f"Length mismatch: {len(queries)} queries, {len(references)} references")

        # Build dataset as list of dicts
        dataset_list = []
        for i in range(n_samples):
            sample = {
                "user_input": queries[i],
                "response": responses[i],
                "retrieved_contexts": contexts[i]
            }

            if references:
                sample["reference"] = references[i]

            if metadata and i < len(metadata):
                sample["metadata"] = metadata[i]

            dataset_list.append(sample)

        # Validate dataset
        validation_report = self.validate_dataset(dataset_list, self.config.metrics)

        if self.config.verbose:
            print(validation_report)

        # Create RAGAS EvaluationDataset
        eval_dataset = EvaluationDataset.from_list(dataset_list)

        return eval_dataset, validation_report

    def evaluate(
        self,
        dataset: EvaluationDataset,
        metrics: Optional[List[str]] = None,
        verbose: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Run RAGAS evaluation on dataset.

        Args:
            dataset: RAGAS EvaluationDataset
            metrics: List of metric names to use (defaults to config.metrics)
            verbose: Override config verbose setting

        Returns:
            Dictionary with results and metadata
        """
        if metrics is None:
            metrics = self.config.metrics

        if verbose is None:
            verbose = self.config.verbose

        # Create metric instances
        metric_instances = []
        for metric_name in metrics:
            if metric_name not in AVAILABLE_METRICS:
                raise ValueError(f"Unknown metric: {metric_name}. Available: {list(AVAILABLE_METRICS.keys())}")

            # Instantiate metric with evaluator LLM
            metric_class = AVAILABLE_METRICS[metric_name]
            try:
                metric = metric_class(llm=self.evaluator_llm)
            except TypeError:
                # Some metrics might not need LLM
                metric = metric_class()

            metric_instances.append(metric)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  RAGAS EVALUATION: {self.config.name}")
            print(f"{'='*60}")
            print(f"\n📊 Dataset: {len(dataset)} samples")
            print(f"📈 Metrics: {', '.join(metrics)}")
            print(f"\n⏳ Running evaluation...")

        # Reset error tracking
        self.sample_errors = []

        # Run evaluation
        try:
            results = evaluate(
                dataset=dataset,
                metrics=metric_instances,
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings
            )

            if verbose:
                print(f"✅ Evaluation complete!\n")
                print(f"{'='*60}")
                print("RESULTS")
                print(f"{'='*60}")
                for metric_name, score in results.items():
                    if metric_name != 'ragas_score':
                        print(f"  {metric_name}: {score:.4f}")
                print(f"{'='*60}\n")

            # Package results with metadata
            return {
                'scores': dict(results),
                'dataset': results.to_pandas(),
                'metadata': self._get_version_metadata(),
                'errors': self.sample_errors,
                'config': self.config.to_dict()
            }

        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            self.sample_errors.append({
                'type': 'evaluation_error',
                'message': error_msg,
                'exception': str(type(e).__name__)
            })

            if self.config.stop_on_error:
                raise
            else:
                print(f"❌ {error_msg}")
                return {
                    'scores': {},
                    'dataset': pd.DataFrame(),
                    'metadata': self._get_version_metadata(),
                    'errors': self.sample_errors,
                    'config': self.config.to_dict()
                }

    def single_sample_evaluate(
        self,
        sample: Dict[str, Any],
        metrics: Optional[List[str]] = None,
        verbose: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample with detailed logging.

        Args:
            sample: Single evaluation sample
            metrics: Metrics to use
            verbose: Verbose output

        Returns:
            Dictionary with scores and metadata
        """
        if verbose is None:
            verbose = self.config.verbose

        # Create single-sample dataset
        dataset = EvaluationDataset.from_list([sample])

        # Run evaluation
        results = self.evaluate(dataset, metrics=metrics, verbose=verbose)

        return results

    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format: str = 'csv'
    ) -> None:
        """
        Export evaluation results with version tracking.

        Args:
            results: Results from evaluate()
            output_path: Path to save results
            format: Export format ('csv', 'json', 'excel')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            # Export sample-level results to CSV
            df = results['dataset']
            df.to_csv(output_path, index=False)

            # Export metadata to companion JSON file
            metadata_path = output_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'scores': results['scores'],
                    'metadata': results['metadata'],
                    'errors': results['errors'],
                    'config': results['config']
                }, f, indent=2)

            print(f"\n✅ Results exported:")
            print(f"   📄 Data: {output_path}")
            print(f"   📋 Metadata: {metadata_path}")

        elif format == 'json':
            # Export everything to JSON
            export_data = {
                'scores': results['scores'],
                'samples': results['dataset'].to_dict(orient='records'),
                'metadata': results['metadata'],
                'errors': results['errors'],
                'config': results['config']
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)

            print(f"\n✅ Results exported to: {output_path}")

        elif format == 'excel':
            # Export to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sample-level scores
                results['dataset'].to_excel(writer, sheet_name='Results', index=False)

                # Summary scores
                summary_df = pd.DataFrame([results['scores']])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Metadata
                metadata_df = pd.DataFrame([results['metadata']])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

            print(f"\n✅ Results exported to: {output_path}")

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv', 'json', or 'excel'")

    def _get_version_metadata(self) -> Dict[str, Any]:
        """Get version metadata for reproducibility."""
        import ragas
        import openai
        import sentence_transformers

        # Try to get optional package versions
        try:
            import google.generativeai as genai
            google_genai_version = genai.__version__
        except:
            google_genai_version = "N/A"

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'ragas_version': ragas.__version__,
            'openai_version': openai.__version__,
            'sentence_transformers_version': sentence_transformers.__version__,
            'google_genai_version': google_genai_version,
            'evaluator_model': f"{self.config.evaluator_llm['provider']}/{self.config.evaluator_llm['model']}",
            'embedding_model': self.config.evaluator_embeddings['model_name']
        }
