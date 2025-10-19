# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Codebase RAG** is a Retrieval-Augmented Generation system for querying codebases using natural language. It indexes code files into a vector database (ChromaDB) and enables semantic search combined with LLM-powered responses about the code.

The system supports 30+ programming languages and file types, using tree-sitter for language-aware parsing and SentenceTransformers for semantic embeddings.

## Environment Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   Copy `.env.example` to `.env` and add at least one API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

   You need at least one of:
   - `OPENAI_API_KEY` - For OpenAI GPT models
   - `GEMINI_API_KEY` - For Google Gemini models

3. **Verify installation:**
   ```bash
   python -c "import chromadb; from src.llm import create_llm_from_env; print('Setup OK')"
   ```

## Common Commands

### Running the RAG System

**Basic usage (auto-detects provider from environment):**
```bash
python main.py -z <path_to_zip> -p "your question about the code"
python main.py -g <github_url> -p "your question about the code"
```

**With specific provider:**
```bash
python main.py -z code.zip -p "How does auth work?" --provider google
python main.py -z code.zip -p "Explain the code" --provider openai
```

**With specific model:**
```bash
python main.py -z code.zip -p "Question?" --model gemini-1.5-pro
python main.py -z code.zip -p "Question?" --model gpt-4o
```

**With temperature control:**
```bash
python main.py -z code.zip -p "Question?" --temperature 0.3
```

**With verbose logging:**
```bash
python main.py -z codebase.zip -p "How does authentication work?" -v
```

**With custom collection name:**
```bash
python main.py -z codebase.zip -p "Explain the architecture" -c my_collection_name
```

### Testing

**Run language detector tests:**
```bash
python tests/test_language_detector.py
```

**Run LLM adapter tests:**
```bash
python tests/test_llm_adapter.py
```

**Run RAGAS evaluator tests:**
```bash
python tests/test_ragas_evaluator.py
```

### RAGAS Evaluation

**Quick evaluation:**
```bash
python evaluate.py \
  --collection-name codeRAG \
  --test-dataset data/evaluation/test_queries.json \
  --metrics faithfulness,answer_relevancy \
  --verbose
```

**Auto-generate test queries:**
```bash
python evaluate.py \
  --collection-name codeRAG \
  --auto-generate 20 \
  --save-dataset data/evaluation/my_queries.json
```

**With reference generation:**
```bash
python evaluate.py \
  --collection-name codeRAG \
  --test-dataset data/evaluation/code_specific_queries.json \
  --generate-references \
  --save-dataset data/evaluation/queries_with_refs.json
```

### Development

This is a Python virtual environment project. The `Lib/` directory contains installed packages.

**Activate environment (if needed):**
```bash
# Windows
.\Scripts\activate

# Unix/MacOS
source Scripts/bin/activate
```

## Architecture

### Four-Phase RAG Pipeline

1. **Data Loading** (`src/text_splitter.py`):
   - Extracts files from ZIP or GitHub repositories
   - Filters files by supported languages using `language_detector.py`
   - Parses code using tree-sitter with language-aware parsers
   - Creates Document objects with metadata (filename, language)

2. **LLM Provider Configuration** (`src/llm/`):
   - Factory pattern for creating LLM providers
   - Auto-detection from environment variables
   - Support for OpenAI and Google Gemini
   - Unified interface across providers

3. **Indexing** (`src/inserter.py`):
   - Generates embeddings using SentenceTransformers (BERT-based)
   - Stores documents with embeddings in ChromaDB
   - Collection management (create/get collections)

4. **Query & Generation** (`src/inserter.py`):
   - Retrieves k=5 most similar code snippets via semantic search
   - Constructs prompt with retrieved context
   - Generates response using configured LLM provider

### Key Components

**`src/inserter.py`** - ChromaCollection class:
- `__init__(collection_name, llm_provider)`: Initialize with optional LLM provider
- `insert_docs(docs)`: Add documents to ChromaDB with automatic embeddings
- `retrieve_k_similar_docs(query, k)`: Semantic search for relevant snippets
- `rag(query, verbose, **llm_kwargs)`: Full RAG pipeline (retrieve + generate)
- `collect_evaluation_data(queries, references)`: Collect data for RAGAS evaluation
- Now uses LLM adapter instead of direct OpenAI client

**`src/text_splitter.py`**:
- `parse_file(filepath, parser_threshold)`: Parse single file into Document chunks
- `load_from_zipfile(zippath)`: Extract and parse all supported files from ZIP
- `load_from_github_link(github_link)`: Download and parse GitHub repository
- Uses temporary directories for extraction, auto-cleans up

**`src/utils/language_detector.py`**:
- `detect_language(filepath)`: Returns language string from file extension
- `is_supported(filepath)`: Check if file type is supported
- `filter_supported_files(paths)`: Filter list to only supported files
- `LANGUAGE_MAP`: Comprehensive mapping of 50+ file extensions to languages

**`src/llm/`** - LLM Adapter Module (COMPLETED):
- **`base.py`**: Abstract `BaseLLMProvider` with `generate()` interface
- **`models.py`**: Enums for `Provider`, `GoogleModel`, `OpenAIModel`
- **`factory.py`**: `LLMFactory` with registry pattern and caching
- **`providers/google_provider.py`**: Google Gemini implementation
- **`providers/openai_provider.py`**: OpenAI GPT implementation
- Unified interface: `generate(messages, temperature, max_tokens, **kwargs) -> LLMResponse`
- Auto-detection from environment with `create_llm_from_env()`
- Message format conversion (OpenAI-style ↔ provider-specific)

**`src/evaluation/`** - RAGAS Evaluation Module (NEW):
- **`ragas_evaluator.py`**: RAGASEvaluator class with metric validation, isolated evaluator LLM, version tracking
- **`test_dataset_generator.py`**: TestDatasetGenerator for creating/loading test queries, ground truth construction
- Supports metrics: faithfulness, answer_relevancy, context_precision, context_recall
- Metric dependency validation (warns about missing references)
- Version metadata for reproducibility
- Error handling with sample-level logging

**`main.py`**:
- CLI entry point with argparse
- Orchestrates the four-phase pipeline
- Provider selection with `--provider` and `--model` flags
- Temperature control with `--temperature`
- Provides verbose logging option for debugging

**`evaluate.py`**:
- Standalone RAGAS evaluation CLI (isolated from main pipeline)
- Supports test dataset loading, auto-generation, reference generation
- Interactive reference curation mode
- Export results with version tracking (CSV/JSON/Excel)

### Data Flow

```
ZIP/GitHub → load_from_zipfile/github → filter by language →
parse with tree-sitter → Document chunks →
ChromaDB (with SentenceTransformer embeddings) →
semantic search (k=5) → augment prompt →
LLM Provider (OpenAI/Google) → generate response
```

### LLM Provider Architecture

The system uses a **Strategy Pattern** with:
- Abstract base class: `BaseLLMProvider`
- Concrete implementations: `GoogleProvider`, `OpenAIProvider`
- Factory for creation: `LLMFactory`
- Registry for extensibility

**Provider auto-detection priority:**
1. Checks for `GEMINI_API_KEY` (prefers Google)
2. Falls back to `OPENAI_API_KEY`
3. Raises error if no keys found

**Message format handling:**
- OpenAI: Native message array support
- Google: Converts message arrays to formatted string prompts
- Both support simple string prompts

### Language Support

The system supports Python, JavaScript/TypeScript, Java, C/C++, Go, Rust, Ruby, PHP, C#, Swift, Kotlin, Scala, and many more. See `src/utils/language_detector.py:14-100` for the complete `LANGUAGE_MAP`.

Files are automatically filtered during loading. Unsupported files are skipped with a logged message.

## Important Notes

### ChromaDB Collections

- Collections are created/retrieved using `_initialize_collection()` in `src/inserter.py:166`
- Collections are **persistent** across runs with the same name
- Use different collection names (`-c` flag) for different codebases
- No built-in collection deletion from CLI - collections accumulate

### Model Configuration

**LLM Providers:**

The system now supports multiple LLM providers via the adapter pattern. Default models:
- **Google Gemini**: `gemini-2.0-flash-exp` (preferred if GEMINI_API_KEY exists)
- **OpenAI**: `gpt-4o-mini` (fallback)

**Programmatic usage:**
```python
from src.llm import create_llm, Provider, GoogleModel
from src.inserter import ChromaCollection

# Explicit provider
llm = create_llm(Provider.GOOGLE, GoogleModel.GEMINI_1_5_PRO)
collection = ChromaCollection('my_code', llm_provider=llm)

# Auto-detect from environment
collection = ChromaCollection('my_code')  # Will use GEMINI_API_KEY or OPENAI_API_KEY

# Custom temperature
collection.rag(query, temperature=0.3)
```

**Available models:**
- **Google**: `gemini-2.0-flash-exp`, `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-1.5-flash-8b`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`

**Embedding model:** SentenceTransformer (default model from library) defined in `inserter.py:23`

To change embedding model, edit:
```python
embedding_function = SentenceTransformerEmbeddingFunction(model_name="your-model-name")
```

### Parser Threshold

The `parser_threshold` parameter (default: 3) in `parse_file()` controls the minimum size of code chunks. Lower values create smaller, more granular fragments. This is passed to LangChain's `LanguageParser`.

### File Processing

- Hidden files (starting with `.`) are automatically skipped
- Hidden directories are excluded from traversal
- Files that fail parsing are caught with try-except and logged, then skipped
- Each parsed chunk gets metadata: `source_file` and `language`

### Verbose Mode

Enable with `-v` flag to see:
- Which files are being parsed (with detected language)
- LLM provider and model being used
- Retrieved code snippets (first 100 chars)
- Context length and fragment count
- Token usage from LLM API

## Known Issues & TODOs

1. **Batch Query Optimization**: The `retrieve_k_similar_docs()` method has a TODO comment about supporting multiple queries at once (see `inserter.py:66-68`).

2. **Collection Management**: No CLI option to delete or list existing collections.

3. **Incremental Updates**: No support for updating existing collections - must recreate entire collection for new files.

4. **Additional Providers**: Easy to extend with more providers (Anthropic Claude, local models, etc.) using the registry pattern in `LLMFactory`.

## Project Context

This is a Deep Learning course project (Universidad de Buenos Aires, 2025) implementing a complete RAG architecture for code understanding. See `CHECKLIST_RAG.md` for the development checklist and `README.md` for full documentation in Spanish.

The main deliverables are:
- Multi-language code parsing with tree-sitter
- Vector-based retrieval with ChromaDB
- Multi-provider LLM integration (OpenAI, Google Gemini)
- Provider-agnostic RAG architecture
- Support for both ZIP files and GitHub repositories

## Recent Updates

**RAGAS Evaluation Module (Completed - NEW):**
- Full RAGAS integration for RAG system evaluation
- Supports 4 core metrics: faithfulness, answer_relevancy, context_precision, context_recall
- **Isolated evaluator LLM** (separate from generation to avoid cache/provider pollution)
- **Metric dependency validation** with warnings for missing ground truth references
- **Version tracking** for reproducibility (Python, RAGAS, models, dependencies)
- Error handling with sample-level logging and detailed reports
- Test dataset generator with ground truth construction utilities
- Interactive reference curation workflow (`--interactive-curation`)
- Standalone CLI (`evaluate.py`) for evaluation without main.py overhead
- Comprehensive test suite in `tests/test_ragas_evaluator.py`
- Sample datasets in `data/evaluation/test_queries.json` and `code_specific_queries.json`
- Export formats: CSV, JSON, Excel with metadata files
- See README.md "Evaluación con RAGAS" section for full documentation

**LLM Adapter Implementation (Completed):**
- Implemented full multi-provider LLM adapter with Strategy + Factory patterns
- Supports both OpenAI (GPT-4, GPT-3.5) and Google Gemini models
- Unified interface via `BaseLLMProvider.generate()`
- Auto-detection from environment variables
- Message format conversion between providers
- CLI integration with `--provider`, `--model`, and `--temperature` flags
- Comprehensive test suite in `tests/test_llm_adapter.py`
- Full backward compatibility with existing code
