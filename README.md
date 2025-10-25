# Codebase RAG - Asistente Inteligente de Código

Un sistema de Retrieval-Augmented Generation (RAG) que permite consultar y comprender bases de código usando lenguaje natural. Indexa archivos de código en una base de datos vectorial y habilita búsqueda semántica con respuestas impulsadas por IA.

## Características Principales

- **Soporte multi-lenguaje**: Maneja más de 30 lenguajes de programación incluyendo Python, JavaScript, TypeScript, Java, C/C++, Go, Rust y más
- **Parseo inteligente de código**: Utiliza parsers específicos por lenguaje (tree-sitter) para extraer fragmentos significativos
- **Recuperación basada en vectores**: Búsqueda por similaridad semántica usando sentence transformers
- **Respuestas impulsadas por IA**: Integración con OpenAI GPT y Google Gemini
- **Soporte ZIP y GitHub**: Procesa codebases desde archivos zip o repositorios GitHub
- **Almacenamiento persistente**: Usa ChromaDB para recuperación eficiente
- **Modo Chat Interactivo**: Conversaciones multi-turno con contexto mantenido

## Arquitectura

### Pipeline RAG

1. **Carga de Datos**: Archivos fuente desde ZIP o GitHub
2. **Preprocesamiento**:
   - Detección automática del lenguaje
   - Filtrado de archivos soportados
   - Parseo específico por lenguaje usando tree-sitter
3. **Text Splitting**: Fragmentación inteligente respetando estructura del lenguaje
4. **Generación de Embeddings**: Conversión a vectores usando SentenceTransformers (BERT)
5. **Almacenamiento**: Embeddings y metadata en ChromaDB
6. **Retrieval**: Búsqueda de los K fragmentos más similares semánticamente
7. **Generación**: GPT/Gemini genera respuesta usando fragmentos relevantes como contexto

### Componentes Principales

1. **Detección de Lenguaje** (`src/utils/language_detector.py`): Identifica tipos de archivo y filtra lenguajes soportados
2. **Fragmentación y Parseo** (`src/text_splitter.py`): Extrae y fragmenta código usando parsers conscientes del lenguaje
3. **Almacenamiento Vectorial y RAG** (`src/inserter.py`): Gestiona ChromaDB, búsqueda por similaridad y generación de respuestas

## Instalación

### Prerequisitos

- Python 3.11 o superior
- OpenAI API key o Gemini API key

### Configuración

```bash
# 1. Clonar repositorio
git clone <repository-url>
cd codebase-rag

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar API key
echo "OPENAI_API_KEY=tu_api_key_aqui" > .env
# O alternativamente:
echo "GEMINI_API_KEY=tu_api_key_aqui" > .env
```

## Guía de Uso

### 💬 Modo Chat Interactivo (Recomendado)

Modo conversacional que mantiene contexto entre preguntas:

```bash
# Desde archivo ZIP
python main.py -z <ruta_al_zip> --config configs/chat_openai.json

# Desde GitHub
python main.py -g <url_github> --config configs/chat_openai.json
```

**Comandos disponibles:**
- `/help`: Mostrar ayuda
- `/clear`: Reiniciar conversación
- `/exit`: Salir del chat

**Configuraciones para chat:**

1. **`chat_openai.json`** (Recomendada): `gpt-4o-mini`, reranking habilitado, balance calidad/costo
2. **`chat_openai_fast.json`**: `gpt-3.5-turbo`, sin reranking, más económica
3. **`chat_openai_premium.json`**: `gpt-4o`, reranking agresivo, máxima calidad

**Nota:** Actualmente solo modelos de OpenAI soportan historial conversacional.

### 📝 Modo Single-Shot

Una pregunta, una respuesta:

```bash
# Desde archivo ZIP
python main.py -z <ruta_al_zip> -p "pregunta sobre el código"

# Desde GitHub
python main.py -g <url_github> -p "pregunta sobre el código"

# Con opciones adicionales
python main.py -z proyecto.zip -p "¿Cómo funciona la autenticación?" \
  -c mi_coleccion --config configs/optimal.json -v
```

**Opciones CLI:**
- `-c, --collection-name`: Nombre de la colección (default: "codeRAG")
- `--config`: Ruta al archivo de configuración JSON
- `-v, --verbose`: Logs detallados (fragmentos recuperados, métricas, tokens)

### Sistema de Configuración JSON

El proyecto incluye 4 configuraciones predefinidas en `configs/`:

1. **`default.json`**: Balance calidad/costo (`gpt-4.1-nano`, k=5, temp=0.1)
2. **`optimal.json`**: Máxima calidad (`gpt-4o`, k=8, embeddings mpnet)
3. **`fast.json`**: Respuestas rápidas (`gpt-3.5-turbo`, k=3, max_tokens=500)
4. **`detailed.json`**: Análisis profundo (`gpt-4.1-nano`, k=7, max_tokens=1500)

**Estructura de configuración:**

```json
{
  "name": "mi_config",
  "prompt": {
    "template": "optimal",
    "max_context_length": 8000
  },
  "model": {
    "name": "gpt-4.1-nano",
    "temperature": 0.1,
    "max_tokens": null
  },
  "retrieval": {
    "k_documents": 5,
    "similarity_threshold": null
  },
  "embeddings": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",
    "distance_function": "l2"
  }
}
```

**Parámetros clave:**

**Prompt:**
- `template`: Plantilla (`default`, `optimal`, `detailed`, `concise`, `spanish`, `beginner_friendly`)
- `max_context_length`: Límite de caracteres del contexto

**Model:**
- `name`: Modelo (`gpt-4o`, `gpt-4.1-nano`, `gpt-3.5-turbo`)
- `temperature`: Aleatoriedad (0.0-2.0, menor = más determinístico)
- `max_tokens`: Límite de tokens en respuesta

**Retrieval:**
- `k_documents`: Número de fragmentos a recuperar
- `similarity_threshold`: Filtro de similitud mínima

**Embeddings:**
- `model_name`: `all-MiniLM-L6-v2` (rápido) o `all-mpnet-base-v2` (mejor calidad)
- `distance_function`: `l2`, `cosine`, o `ip`

**Crear configuración personalizada:**

```bash
cp configs/default.json configs/mi_config.json
# Editar mi_config.json según necesidades
python main.py -z code.zip -p "pregunta" --config configs/mi_config.json
```

### Uso Programático

```python
from src.config_loader import RAGConfig
from src.inserter import ChromaCollection
from src.text_splitter import load_from_zipfile

# Cargar y procesar codebase
docs = load_from_zipfile('codebase.zip')

# Inicializar colección con configuración
config = RAGConfig.from_json("configs/optimal.json")
collection = ChromaCollection('mi_proyecto', config=config)
collection.insert_docs(docs)

# Consultar codebase
respuesta = collection.rag("¿Cómo funciona la autenticación?")
print(respuesta)

# Recuperar documentos similares
docs_similares, _ = collection.retrieve_k_similar_docs(["función de login"], k=5)
```

### Lenguajes Soportados

**Programación:** Python, JavaScript/TypeScript, Java, C/C++, Go, Rust, Ruby, PHP, C#, Swift, Kotlin, Scala, Haskell, Lua, LaTeX

**Documentación:** Markdown, JSON, YAML, TOML, XML, TXT, RST, INI, CFG

**Web:** HTML, CSS/SCSS/SASS/LESS

**Otros:** Shell scripts, SQL, R

Ver lista completa de extensiones en `src/utils/language_detector.py`.

## Evaluación con RAGAS

El sistema incluye evaluación basada en [RAGAS](https://docs.ragas.io/) para medir calidad del RAG.

### Métricas Disponibles

- **Faithfulness**: Respuestas basadas en contexto recuperado (sin alucinaciones)
- **Answer Relevancy**: Relevancia de la respuesta para la pregunta
- **Context Precision**: Precisión del contexto recuperado (requiere referencias)
- **Context Recall**: Completitud del contexto (requiere referencias)

### Evaluación Rápida

```bash
# 1. Preparar dataset de prueba (ver data/evaluation/test_queries.json)
# 2. Ejecutar evaluación
python evaluate.py \
  --collection-name codeRAG \
  --test-dataset data/evaluation/test_queries.json \
  --metrics faithfulness,answer_relevancy

# 3. Ver resultados en results/evaluation_results.csv
```

### Opciones Avanzadas

```bash
# Auto-generar queries de prueba
python evaluate.py --collection-name codeRAG --auto-generate 20 \
  --save-dataset data/evaluation/my_queries.json

# Generar referencias automáticamente
python evaluate.py --test-dataset queries.json --generate-references \
  --save-dataset queries_with_refs.json

# Curación interactiva
python evaluate.py --test-dataset queries.json --interactive-curation
```

### Formato del Dataset

```json
{
  "samples": [
    {
      "question": "¿Cómo funciona la autenticación?",
      "reference": "La autenticación se maneja mediante...",
      "metadata": {"category": "security", "difficulty": "medium"}
    }
  ]
}
```

### Interpretando Resultados

**Rangos de puntuación (0.0 - 1.0):**
- **0.8 - 1.0**: Excelente
- **0.6 - 0.8**: Bueno
- **0.4 - 0.6**: Regular - Necesita optimización
- **< 0.4**: Pobre - Requiere cambios significativos

Ver `configs/evaluation.json` para configurar modelo evaluador, métricas y batch size.

## Datasets de Prueba

El directorio `dataset/` contiene proyectos de prueba:

1. **SMTP-Protos** (~200KB): Protocolo de red en C - pruebas rápidas
2. **jam-py** (~15MB): Framework web JavaScript/Python - parseo multi-lenguaje
3. **NumPy** (~10MB): Librería científica Python/C - código científico complejo

```bash
python main.py -z dataset/numpy-main.zip -p "¿Qué hace la función array?"
```

## Estructura del Proyecto

```
codebase-rag/
├── src/
│   ├── inserter.py              # ChromaDB e implementación RAG
│   ├── text_splitter.py         # Parseo de archivos
│   ├── config_loader.py         # Cargador de configs
│   ├── prompt_loader.py         # Plantillas de prompts
│   └── utils/
│       └── language_detector.py # Detección de lenguaje
├── configs/                      # Configuraciones JSON
├── prompts/                      # Plantillas de prompts
├── dataset/                      # Datasets de prueba
├── data/evaluation/              # Queries de evaluación
├── results/                      # Resultados de evaluación
├── main.py                       # CLI principal
├── evaluate.py                   # CLI de evaluación
└── requirements.txt
```

## Testing

```bash
# Tests de detección de lenguaje
python tests/test_language_detector.py

# Tests de evaluador RAGAS
python tests/test_ragas_evaluator.py
```

## Dependencias Principales

- `chromadb`: Base de datos vectorial
- `openai`: Integración con GPT
- `langchain`: Procesamiento de documentos
- `sentence-transformers`: Generación de embeddings
- `tree-sitter`: Parseo consciente del lenguaje
- `ragas`: Framework de evaluación RAG

Ver [requirements.txt](requirements.txt) para la lista completa.

## Limitaciones Actuales

- Requiere API key de OpenAI o Gemini (costos por uso)
- Archivos binarios e imágenes no soportados
- Rendimiento depende del tamaño de la codebase
- Calidad depende de la relevancia de fragmentos recuperados

## Mejoras Futuras

- Embeddings especializados en código
- Soporte para lenguajes de programación adicionales
- Metadata filtering avanzado
- Agentic RAG
- Escalabilidad a repos > 100K docs

## Licencia

Este proyecto se proporciona tal cual para fines educativos y de desarrollo.
