"""
Test Dataset Generator for RAGAS Evaluation.

Provides utilities for:
- Loading test queries from JSON
- Generating test queries from codebase
- Building reference answers (ground truth)
- Interactive reference curation
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class TestQuery:
    """Represents a test query with optional reference."""
    question: str
    reference: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"question": self.question}
        if self.reference:
            result["reference"] = self.reference
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class TestDatasetGenerator:
    """
    Generate and manage test datasets for RAG evaluation.

    Example:
        >>> generator = TestDatasetGenerator()
        >>> queries = generator.load_from_json("data/evaluation/test_queries.json")
        >>> # Or auto-generate from codebase
        >>> queries = generator.generate_from_codebase(collection, num_samples=10)
    """

    # Template questions for code understanding
    CODE_QUESTION_TEMPLATES = [
        "How does {topic} work in this codebase?",
        "What is the purpose of {component}?",
        "Explain the implementation of {feature}.",
        "How is {concept} handled?",
        "What are the main components of {system}?",
        "How does the {module} module work?",
        "What patterns are used in {area}?",
        "How is data flow managed in {component}?",
        "What testing approaches are used for {feature}?",
        "How is error handling implemented in {module}?"
    ]

    def __init__(self):
        """Initialize test dataset generator."""
        pass

    def load_from_json(self, file_path: str) -> List[TestQuery]:
        """
        Load test queries from JSON file.

        Expected format:
        {
          "samples": [
            {
              "question": "How does auth work?",
              "reference": "Authentication is handled by...",
              "metadata": {"category": "security"}
            }
          ]
        }

        Args:
            file_path: Path to JSON file

        Returns:
            List of TestQuery objects
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Test dataset not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'samples' not in data:
            raise ValueError("JSON must contain 'samples' key")

        queries = []
        for sample in data['samples']:
            query = TestQuery(
                question=sample['question'],
                reference=sample.get('reference'),
                metadata=sample.get('metadata')
            )
            queries.append(query)

        return queries

    def save_to_json(self, queries: List[TestQuery], file_path: str) -> None:
        """
        Save test queries to JSON file.

        Args:
            queries: List of TestQuery objects
            file_path: Path to save JSON file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "samples": [q.to_dict() for q in queries],
            "metadata": {
                "num_samples": len(queries),
                "with_references": sum(1 for q in queries if q.reference),
                "without_references": sum(1 for q in queries if not q.reference)
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(queries)} queries to {file_path}")

    def generate_from_codebase(
        self,
        collection,
        num_samples: int = 10,
        topics: Optional[List[str]] = None
    ) -> List[TestQuery]:
        """
        Auto-generate test queries from codebase analysis.

        This uses simple heuristics to generate questions. For better quality,
        consider using an LLM to generate questions based on the codebase.

        Args:
            collection: ChromaCollection instance
            num_samples: Number of queries to generate
            topics: Optional list of topics/components to ask about

        Returns:
            List of TestQuery objects (without references)
        """
        if topics is None:
            # Try to infer topics from collection metadata
            # This is a simplified version - could be enhanced with actual codebase analysis
            topics = [
                "authentication", "database", "API", "configuration",
                "error handling", "testing", "data processing", "routing",
                "validation", "logging", "caching", "security"
            ]

        queries = []
        templates_used = set()

        for _ in range(num_samples):
            # Randomly select template and topic
            template = random.choice(self.CODE_QUESTION_TEMPLATES)
            topic = random.choice(topics)

            # Avoid duplicate template-topic combinations if possible
            combo = (template, topic)
            if combo in templates_used and len(templates_used) < len(self.CODE_QUESTION_TEMPLATES) * len(topics):
                continue

            templates_used.add(combo)

            # Generate question
            question = template.format(
                topic=topic,
                component=topic,
                feature=topic,
                concept=topic,
                system=topic,
                module=topic,
                area=topic
            )

            queries.append(TestQuery(
                question=question,
                metadata={"category": "auto_generated", "topic": topic}
            ))

        return queries

    def build_reference_answers(
        self,
        queries: List[TestQuery],
        collection,
        verbose: bool = False
    ) -> List[TestQuery]:
        """
        Generate reference answers using the RAG system.

        NOTE: This generates *candidate* references that should be manually reviewed.
        The quality of auto-generated references depends on the RAG system quality.

        Args:
            queries: List of TestQuery objects
            collection: ChromaCollection instance with RAG
            verbose: Print progress

        Returns:
            List of TestQuery objects with generated references
        """
        if verbose:
            print(f"\n⏳ Generating reference answers for {len(queries)} queries...")
            print("⚠️  Note: Auto-generated references should be manually reviewed!")

        queries_with_refs = []

        for i, query in enumerate(queries, 1):
            if verbose:
                print(f"\n[{i}/{len(queries)}] Processing: {query.question[:60]}...")

            # Generate reference using RAG
            # Use lower temperature for more deterministic responses
            try:
                reference, _ = collection.rag(
                    query=query.question,
                    verbose=False
                )

                # Create new TestQuery with reference
                new_query = TestQuery(
                    question=query.question,
                    reference=reference,
                    metadata={
                        **(query.metadata or {}),
                        "reference_source": "auto_generated",
                        "confidence": "needs_review"
                    }
                )

                queries_with_refs.append(new_query)

                if verbose:
                    print(f"   ✅ Generated ({len(reference)} chars)")

            except Exception as e:
                if verbose:
                    print(f"   ❌ Error: {str(e)}")

                # Keep query without reference
                queries_with_refs.append(query)

        if verbose:
            successful = sum(1 for q in queries_with_refs if q.reference)
            print(f"\n✅ Generated {successful}/{len(queries)} references")

        return queries_with_refs

    def curate_references(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
        resume: bool = True
    ) -> None:
        """
        Interactive CLI for manual reference creation/curation.

        Args:
            dataset_path: Path to existing dataset JSON
            output_path: Path to save curated dataset (defaults to dataset_path)
            resume: If True, skip queries that already have references
        """
        if output_path is None:
            output_path = dataset_path

        # Load existing queries
        queries = self.load_from_json(dataset_path)

        print(f"\n{'='*60}")
        print("INTERACTIVE REFERENCE CURATION")
        print(f"{'='*60}")
        print(f"\n📋 Loaded {len(queries)} queries from {dataset_path}")

        if resume:
            queries_to_curate = [q for q in queries if not q.reference]
            print(f"📝 {len(queries_to_curate)} queries need references (resume mode)")
        else:
            queries_to_curate = queries
            print(f"📝 Curating all {len(queries_to_curate)} queries")

        print("\nInstructions:")
        print("  • Enter reference answer for each question")
        print("  • Press Enter to skip")
        print("  • Type 'quit' to save and exit")
        print(f"{'='*60}\n")

        curated = 0

        for i, query in enumerate(queries_to_curate, 1):
            print(f"\n[{i}/{len(queries_to_curate)}] Question:")
            print(f"  {query.question}")

            if query.reference:
                print(f"\nCurrent reference:")
                print(f"  {query.reference[:200]}...")

            print(f"\nEnter reference answer (or 'skip'/'quit'):")
            user_input = input("> ").strip()

            if user_input.lower() == 'quit':
                print("\n💾 Saving and exiting...")
                break
            elif user_input.lower() == 'skip' or not user_input:
                print("⏭️  Skipped")
                continue
            else:
                # Update reference
                query.reference = user_input
                if query.metadata is None:
                    query.metadata = {}
                query.metadata["reference_source"] = "manual_curation"
                curated += 1
                print("✅ Reference saved")

        # Save curated dataset
        self.save_to_json(queries, output_path)

        print(f"\n{'='*60}")
        print(f"✅ Curation complete!")
        print(f"   • Curated: {curated} references")
        print(f"   • Saved to: {output_path}")
        print(f"{'='*60}\n")

    def validate_dataset(
        self,
        queries: List[TestQuery],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Validate test dataset for use with specified metrics.

        Args:
            queries: List of TestQuery objects
            metrics: List of metric names

        Returns:
            Validation report dictionary
        """
        from .ragas_evaluator import METRIC_REQUIREMENTS

        report = {
            "valid": True,
            "warnings": [],
            "stats": {
                "total_queries": len(queries),
                "with_references": sum(1 for q in queries if q.reference),
                "without_references": sum(1 for q in queries if not q.reference)
            }
        }

        # Check if any metrics require references
        needs_reference = False
        for metric in metrics:
            if metric in METRIC_REQUIREMENTS:
                if METRIC_REQUIREMENTS[metric]["needs_reference"]:
                    needs_reference = True
                    break

        if needs_reference and report["stats"]["without_references"] > 0:
            report["warnings"].append(
                f"⚠️  {report['stats']['without_references']} queries missing references. "
                f"Some metrics (context_precision, context_recall) require references."
            )

        # Check for empty questions
        empty_questions = sum(1 for q in queries if not q.question.strip())
        if empty_questions > 0:
            report["valid"] = False
            report["warnings"].append(f"❌ {empty_questions} queries have empty questions")

        return report
