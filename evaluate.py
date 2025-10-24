"""
Standalone RAGAS Evaluation Script for Codebase RAG.

This script provides a CLI for evaluating RAG system performance using RAGAS metrics.
It is completely isolated from the main RAG pipeline to avoid performance overhead.

Usage:
    python evaluate.py --collection-name codeRAG --test-dataset data/evaluation/test_queries.json
    python evaluate.py --collection-name codeRAG --auto-generate 20 --metrics faithfulness,answer_relevancy
    python evaluate.py --collection-name codeRAG --test-dataset queries.json --generate-references
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import RAGASEvaluator, EvaluationConfig, TestDatasetGenerator
from src.inserter import ChromaCollection
from src.config_loader import RAGConfig


def main():
    parser = argparse.ArgumentParser(
        prog="RAGAS Evaluation",
        description="Evaluate Codebase RAG system using RAGAS metrics"
    )

    # Required arguments
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Name of the ChromaDB collection to evaluate"
    )

    # Test dataset options (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--test-dataset",
        help="Path to test queries JSON file"
    )
    dataset_group.add_argument(
        "--auto-generate",
        type=int,
        metavar="N",
        help="Auto-generate N test queries from codebase"
    )

    # Configuration
    parser.add_argument(
        "--eval-config",
        default="configs/evaluation.json",
        help="Path to evaluation config JSON (default: configs/evaluation.json)"
    )

    parser.add_argument(
        "--rag-config",
        help="Path to RAG config JSON for generation model (optional)"
    )

    # Metrics
    parser.add_argument(
        "--metrics",
        help="Comma-separated list of metrics (overrides eval config). "
             "Available: faithfulness, answer_relevancy, context_precision, context_recall"
    )

    # Reference handling
    parser.add_argument(
        "--generate-references",
        action="store_true",
        help="Auto-generate reference answers using RAG (requires manual review)"
    )

    parser.add_argument(
        "--interactive-curation",
        action="store_true",
        help="Interactively curate/add references to test dataset"
    )

    # Output
    parser.add_argument(
        "--output",
        default="results/evaluation_results.csv",
        help="Output path for results (default: results/evaluation_results.csv)"
    )

    parser.add_argument(
        "--format",
        choices=["csv", "json", "excel"],
        default="csv",
        help="Output format (default: csv)"
    )

    # Behavior
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with detailed logging"
    )

    parser.add_argument(
        "--save-dataset",
        help="Save generated/processed test dataset to this path"
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*60)
    print("  RAGAS EVALUATION FOR CODEBASE RAG")
    print("="*60)

    # ===================================================================
    # PHASE 1: Load Configurations
    # ===================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: LOADING CONFIGURATIONS")
    print(f"{'='*60}\n")

    # Load evaluation config
    try:
        eval_config = EvaluationConfig.from_json(args.eval_config)
        print(f"✅ Evaluation config loaded: {eval_config.name}")
        print(f"   {eval_config.description}")

        # Override metrics if specified
        if args.metrics:
            eval_config.metrics = [m.strip() for m in args.metrics.split(",")]
            print(f"   Overriding metrics: {', '.join(eval_config.metrics)}")

        # Override verbose if specified
        if args.verbose:
            eval_config.verbose = True

    except Exception as e:
        print(f"❌ Error loading evaluation config: {e}")
        sys.exit(1)

    # Load RAG config if specified
    rag_config = None
    if args.rag_config:
        try:
            rag_config = RAGConfig.from_json(args.rag_config)
            print(f"✅ RAG config loaded: {rag_config.name}")
        except Exception as e:
            print(f"❌ Error loading RAG config: {e}")
            sys.exit(1)

    # ===================================================================
    # PHASE 2: Load Existing Collection
    # ===================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: LOADING EXISTING CHROMADB COLLECTION")
    print(f"{'='*60}\n")

    print(f"📊 Colección: '{args.collection_name}'")
    try:
        if rag_config:
            print(f"⏳ Cargando colección existente con config '{rag_config.name}'...")
            print(f"   • Prompt: {rag_config.prompt.template}")
            print(f"   • Modelo: {rag_config.model.name}")
            print(f"   • Embeddings: {rag_config.embeddings.model_name}")
            print(f"   • Retrieval: {rag_config.retrieval.k_documents} documentos")
            if rag_config.rerank.enabled:
                print(f"   • Reranking: {rag_config.rerank.strategy} (retrieve {rag_config.rerank.retrieve_k} → top {rag_config.rerank.top_n})")
            else:
                print(f"   • Reranking: Deshabilitado")
            collection = ChromaCollection(args.collection_name, config=rag_config)
        else:
            print(f"⏳ Cargando colección existente con configuración por defecto...")
            collection = ChromaCollection(args.collection_name)

        print(f"✅ Colección cargada correctamente")

        # Verificar que la colección tiene documentos
        collection_size = collection.chroma_collection.count()
        if collection_size == 0:
            print(f"\n⚠️  Advertencia: La colección '{args.collection_name}' está vacía")
            print(f"   Debe indexar documentos primero usando:")
            print(f"   python index.py -z <archivo.zip> -c {args.collection_name}")
            sys.exit(1)

        print(f"   • Documentos en colección: {collection_size}")

    except Exception as e:
        print(f"❌ Error: No se pudo acceder a la colección '{args.collection_name}'")
        print(f"   {e}")
        print(f"\n   ¿La colección existe? Use index.py para crearla primero:")
        print(f"   python index.py -z <archivo.zip> -c {args.collection_name}")
        sys.exit(1)

    # ===================================================================
    # PHASE 3: Prepare Test Dataset
    # ===================================================================
    print(f"\n{'='*60}")
    print("PHASE 3: PREPARING TEST DATASET")
    print(f"{'='*60}\n")

    generator = TestDatasetGenerator()

    # Load or generate test queries
    if args.test_dataset:
        print(f"📂 Loading test dataset from: {args.test_dataset}")
        try:
            test_queries = generator.load_from_json(args.test_dataset)
            print(f"✅ Loaded {len(test_queries)} test queries")
        except Exception as e:
            print(f"❌ Error loading test dataset: {e}")
            sys.exit(1)

    elif args.auto_generate:
        print(f"🔧 Auto-generating {args.auto_generate} test queries...")
        try:
            test_queries = generator.generate_from_codebase(
                collection=collection,
                num_samples=args.auto_generate
            )
            print(f"✅ Generated {len(test_queries)} test queries")

            # Optionally save generated queries
            if args.save_dataset:
                generator.save_to_json(test_queries, args.save_dataset)

        except Exception as e:
            print(f"❌ Error generating queries: {e}")
            sys.exit(1)

    # Interactive curation mode
    if args.interactive_curation:
        print(f"\n🖊️  Starting interactive reference curation...")
        if args.test_dataset:
            dataset_path = args.test_dataset
        elif args.save_dataset:
            # Save generated queries first
            generator.save_to_json(test_queries, args.save_dataset)
            dataset_path = args.save_dataset
        else:
            # Save to temp file
            dataset_path = "temp_queries_for_curation.json"
            generator.save_to_json(test_queries, dataset_path)

        generator.curate_references(dataset_path)

        # Reload curated queries
        test_queries = generator.load_from_json(dataset_path)

    # Generate references if requested
    elif args.generate_references:
        print(f"\n🤖 Auto-generating reference answers...")
        print("⚠️  Note: Auto-generated references should be manually reviewed!")

        test_queries = generator.build_reference_answers(
            test_queries,
            collection=collection,
            verbose=args.verbose
        )

        # Save if requested
        if args.save_dataset:
            generator.save_to_json(test_queries, args.save_dataset)

    # Validate dataset
    print(f"\n📋 Validating dataset for metrics: {', '.join(eval_config.metrics)}")
    validation = generator.validate_dataset(test_queries, eval_config.metrics)

    if validation["warnings"]:
        for warning in validation["warnings"]:
            print(f"   {warning}")

    # ===================================================================
    # PHASE 4: Collect Evaluation Data
    # ===================================================================
    print(f"\n{'='*60}")
    print("PHASE 4: COLLECTING EVALUATION DATA")
    print(f"{'='*60}\n")

    # Extract queries and references (keep partial references, don't drop)
    queries = [q.question for q in test_queries]
    references = [q.reference if q.reference else None for q in test_queries]

    # Count references for reporting
    ref_count = sum(1 for r in references if r is not None)
    if ref_count < len(queries):
        print(f"   ⚠️  Partial references: {ref_count}/{len(queries)} queries have ground truth")
        print(f"   Reference-requiring metrics will only evaluate on {ref_count} samples")

    print(f"⏳ Running {len(queries)} queries through RAG pipeline...")

    try:
        eval_samples = collection.collect_evaluation_data(
            queries=queries,
            references=references,
            verbose=args.verbose
        )

        if not eval_samples:
            print("❌ No evaluation samples collected. Check your queries and collection.")
            sys.exit(1)

        print(f"✅ Collected {len(eval_samples)} evaluation samples")

    except Exception as e:
        print(f"❌ Error collecting evaluation data: {e}")
        sys.exit(1)

    # ===================================================================
    # PHASE 5: Run RAGAS Evaluation
    # ===================================================================
    print(f"\n{'='*60}")
    print("PHASE 5: RUNNING RAGAS EVALUATION")
    print(f"{'='*60}\n")

    try:
        # Initialize evaluator
        evaluator = RAGASEvaluator(eval_config)

        # Prepare dataset for RAGAS
        from ragas import EvaluationDataset
        ragas_dataset = EvaluationDataset.from_list(eval_samples)

        # Run evaluation
        results = evaluator.evaluate(
            dataset=ragas_dataset,
            metrics=eval_config.metrics,
            verbose=args.verbose
        )

        # ===================================================================
        # PHASE 6: Export Results
        # ===================================================================
        print(f"\n{'='*60}")
        print("PHASE 6: EXPORTING RESULTS")
        print(f"{'='*60}\n")

        evaluator.export_results(
            results=results,
            output_path=args.output,
            format=args.format
        )

        # ===================================================================
        # Summary
        # ===================================================================
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}\n")

        print(f"📊 Metrics ({len(eval_config.metrics)}):")
        for metric, score in results['scores'].items():
            if metric != 'ragas_score':
                print(f"   • {metric}: {score:.4f}")

        print(f"\n📁 Results saved to: {args.output}")

        if results['errors']:
            print(f"\n⚠️  {len(results['errors'])} errors occurred during evaluation")
            print("   Check the exported results for details")

        print(f"\n{'='*60}")
        print("✅ EVALUATION COMPLETE")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
