# Codebase RAG - Asistente Inteligente de Código

Un sistema de Retrieval-Augmented Generation (RAG) diseñado para ayudarte a consultar y comprender bases de código usando lenguaje natural. Esta herramienta indexa archivos de código en una base de datos vectorial y habilita búsqueda semántica y respuestas impulsadas por IA sobre tu codebase.

## Descripción del Proyecto

Codebase RAG es un asistente de codebase que permite al usuario preguntar al agente usando lenguaje natural y obtener referencias y explicaciones sobre el código. El proyecto está diseñado para usarse con codebases relativamente grandes, ayudando a los desarrolladores a tener un conocimiento más completo y basado en la totalidad del código sobre los repositorios donde se encuentren trabajando.

### Características Principales

- **Soporte multi-lenguaje**: Maneja más de 30 lenguajes de programación y tipos de archivo incluyendo Python, JavaScript, TypeScript, Java, C/C++, Go, Rust y más
- **Parseo inteligente de código**: Utiliza parsers específicos por lenguaje para extraer fragmentos de código significativos
- **Recuperación basada en vectores**: Aprovecha sentence transformers para búsqueda por similaridad semántica
- **Respuestas impulsadas por IA**: Integración con modelos GPT de OpenAI para proporcionar respuestas en lenguaje natural
- **Soporte de archivos ZIP**: Procesa codebases completos empaquetados como archivos zip
- **Almacenamiento persistente**: Usa ChromaDB para almacenamiento y recuperación eficiente de documentos

### Input del Sistema

Texto en lenguaje natural sobre la codebase. Por ejemplo:
- "¿Cómo funciona el sistema de autenticación?"
- "¿Qué tests le faltan a esta función?"
- "¿Cuál es la relación entre main.py y models.py?"
- "Explicame la arquitectura del proyecto"

### Output del Sistema

Snippets de código (con sus archivos de referencia) acompañados de explicaciones contextuales que responden a las preguntas del usuario.

## Arquitectura

El sistema consta de tres componentes principales:

1. **Detección de Lenguaje** ([src/utils/language_detector.py](src/utils/language_detector.py)): Identifica tipos de archivo y filtra lenguajes soportados
2. **Fragmentación y Parseo de Texto** ([src/text_splitter.py](src/text_splitter.py)): Extrae y fragmenta código de archivos fuente usando parsers conscientes del lenguaje
3. **Almacenamiento Vectorial y RAG** ([src/inserter.py](src/inserter.py)): Gestiona colecciones de ChromaDB, realiza búsqueda por similaridad y genera respuestas de IA

### Pipeline RAG

1. **Carga de Datos**: Los archivos fuente se cargan desde archivos ZIP
2. **Preprocesamiento**:
   - Detección automática del lenguaje de programación
   - Filtrado de archivos soportados
   - Parseo específico por lenguaje usando tree-sitter
3. **Text Splitting**: Fragmentación inteligente del código respetando la estructura del lenguaje
4. **Generación de Embeddings**: Conversión de fragmentos de código a vectores usando SentenceTransformers (BERT)
5. **Almacenamiento**: Embeddings y metadata se almacenan en ChromaDB
6. **Retrieval**: Búsqueda de los K fragmentos más similares semánticamente a la consulta
7. **Generación**: Los fragmentos relevantes se pasan a GPT junto con la pregunta para generar una respuesta contextual

## Instalación

### Prerequisitos

- Python 3.11 o superior
- OpenAI API key o Gemini API key

### Configuración

1. Clona el repositorio:
```bash
git clone <repository-url>
cd codebase-rag
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Crea un archivo `.env` en la raíz del proyecto y agrega tu API key de OpenAI:
```bash
OPENAI_API_KEY=tu_api_key_aqui
```

## Guía de Uso

El sistema soporta **dos modos de operación**:

1. **Modo Chat Interactivo** (nuevo): Conversaciones multi-turno con contexto mantenido
2. **Modo Single-Shot**: Una pregunta, una respuesta y salida (modo original)

---

### 💬 Modo Chat Interactivo (Recomendado)

Modo conversacional que mantiene el contexto entre preguntas. Ideal para exploración de código.

#### Desde archivo ZIP:
```bash
python main.py -z <ruta_al_zip> --config configs/chat_openai.json
```

**Ejemplo:**
```bash
python main.py -z test_data --config configs/chat_openai.json
```

**Sesión ejemplo:**
```
====================================
  MODO CHAT INTERACTIVO
====================================

Config: chat_openai
Modelo: gpt-4o-mini

Comandos: /help, /clear, /exit

====================================

Q: ¿Cómo se crea un usuario?
Buscando...
A: Para crear un usuario, instancia la clase Usuario...

Q: ¿Y cómo le asigno un rol?
Buscando...
A: (Recuerda el contexto anterior sobre usuarios)
Los roles están definidos en config.py...

Q: /clear
Conversación reiniciada

Q: /exit
Hasta pronto!
```

#### Desde repositorio de GitHub:
```bash
python main.py -g <url_github> --config configs/chat_openai.json
```

#### Comandos disponibles en modo chat:

| Comando | Descripción |
|---------|-------------|
| `/help` | Mostrar ayuda de comandos |
| `/clear` | Reiniciar conversación (nuevo contexto) |
| `/exit` | Salir del chat |

#### Configuraciones para modo chat:

El sistema incluye configuraciones optimizadas para chat con OpenAI (soporta historial conversacional):

**1. `chat_openai.json` - Balanceada** (Recomendada)
```bash
python main.py -z proyecto.zip --config configs/chat_openai.json
```
- Modelo: `gpt-4o-mini` (balance calidad/costo)
- Reranking: Cross-Encoder habilitado
- Historial: ✅ Completo

**2. `chat_openai_fast.json` - Rápida y económica**
```bash
python main.py -z proyecto.zip --config configs/chat_openai_fast.json
```
- Modelo: `gpt-3.5-turbo` (más barato)
- Sin reranking (más rápido)
- Historial: ✅ Completo

**3. `chat_openai_premium.json` - Máxima calidad**
```bash
python main.py -z proyecto.zip --config configs/chat_openai_premium.json
```
- Modelo: `gpt-4o` (el mejor)
- Reranking agresivo (25→8 docs)
- Historial: ✅ Completo

**Nota:** Solo OpenAI soporta historial conversacional. Con Gemini cada pregunta es independiente.

Ver documentación completa: [configs/CHAT_CONFIGS.md](configs/CHAT_CONFIGS.md)

---

### 📝 Modo Single-Shot

Una pregunta, una respuesta. Útil para scripts y automatización.

#### Desde archivo ZIP:
```bash
python main.py -z <ruta_al_zip> -p "pregunta sobre el código"
```

**Ejemplo:**
```bash
python main.py -z test_codebase.zip -p "¿Cómo funciona la autenticación?"
```

#### Desde repositorio de GitHub:
```bash
python main.py -g <url_github> -p "pregunta sobre el código"
```

**Ejemplo:**
```bash
python main.py -g https://github.com/usuario/repo -p "¿Qué hace este código?"
```

#### Opciones adicionales:

- `-c, --collection-name`: Nombre de la colección (default: "codeRAG")
- `--config`: Ruta al archivo de configuración JSON (opcional)
- `-v, --verbose`: Muestra logs detallados del proceso RAG

**Ejemplo con todas las opciones:**
```bash
python main.py -z mi_proyecto.zip -p "¿Cómo se crea un usuario?" -c mi_coleccion --config configs/optimal.json -v
```

### Modo Verbose

El flag `-v` o `--verbose` activa logs detallados que muestran:

1. **Fragmentos recuperados**: Preview de los 5 fragmentos más relevantes encontrados
2. **Información del contexto**: Longitud total y número de fragmentos usados
3. **Detalles de generación**: Modelo usado, tokens aproximados y tokens totales consumidos

**Sin verbose:**
```bash
python main.py -z test_codebase.zip -p "¿Qué hace este código?"
# Solo muestra: Fases principales → Respuesta final
```

**Con verbose:**
```bash
python main.py -z test_codebase.zip -p "¿Qué hace este código?" -v
# Muestra: Fases + Fragmentos recuperados + Métricas + Respuesta
```

El modo verbose es útil para:
- Entender qué fragmentos de código está usando el RAG
- Debuggear respuestas inesperadas
- Evaluar la calidad de la recuperación
- Ver el consumo de tokens de la API

### Sistema de Configuración JSON

El sistema permite configurar todos los parámetros del RAG mediante archivos JSON, facilitando la experimentación con diferentes configuraciones sin modificar código.

#### Archivos de Configuración Incluidos

El proyecto incluye 4 configuraciones predefinidas en la carpeta `configs/`:

**1. `default.json` - Configuración balanceada**
```bash
python main.py -z code.zip -p "pregunta" --config configs/default.json
```
- Modelo: `gpt-4.1-nano`
- K documentos: 5
- Temperature: 0.1
- Uso: General, balance entre calidad y costo

**2. `optimal.json` - Máxima calidad**
```bash
python main.py -z code.zip -p "pregunta" --config configs/optimal.json
```
- Modelo: `gpt-4o`
- K documentos: 8
- Temperature: 0.05
- Embeddings: `all-mpnet-base-v2` (mejor calidad)
- Uso: Análisis crítico, máxima precisión

**3. `fast.json` - Respuestas rápidas**
```bash
python main.py -z code.zip -p "pregunta" --config configs/fast.json
```
- Modelo: `gpt-3.5-turbo`
- K documentos: 3
- Max tokens: 500
- Uso: Consultas rápidas, desarrollo

**4. `detailed.json` - Explicaciones exhaustivas**
```bash
python main.py -z code.zip -p "pregunta" --config configs/detailed.json
```
- Modelo: `gpt-4.1-nano`
- K documentos: 7
- Max tokens: 1500
- Uso: Análisis profundo, documentación

#### Estructura de un Archivo de Configuración

```json
{
  "name": "mi_config",
  "description": "Descripción de la configuración",

  "prompt": {
    "template": "optimal",
    "include_metadata": true,
    "max_context_length": 8000
  },

  "model": {
    "name": "gpt-4.1-nano",
    "temperature": 0.1,
    "max_tokens": null,
    "top_p": 1.0
  },

  "retrieval": {
    "k_documents": 5,
    "include_metadata": true,
    "similarity_threshold": null
  },

  "text_splitting": {
    "parser_threshold": 3
  },

  "embeddings": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",
    "normalize_embeddings": true,
    "distance_function": "l2"
  }
}
```

#### Parámetros Configurables

**Prompt:**
- `template`: Plantilla de prompt (`default`, `optimal`, `detailed`, `concise`, `spanish`, `beginner_friendly`)
- `max_context_length`: Límite de caracteres del contexto

**Model (OpenAI):**
- `name`: Modelo a usar (`gpt-4o`, `gpt-4.1-nano`, `gpt-3.5-turbo`)
- `temperature`: Controla aleatoriedad (0.0-2.0, menor = más determinístico)
- `max_tokens`: Límite de tokens en respuesta (null = sin límite)
- `top_p`: Nucleus sampling (0.0-1.0)

**Retrieval:**
- `k_documents`: Número de fragmentos a recuperar
- `similarity_threshold`: Filtro de similitud mínima (null = sin filtro)

**Text Splitting:**
- `parser_threshold`: Tamaño mínimo de fragmentos de código

**Embeddings:**
- `model_name`: Modelo de sentence-transformers
  - `all-MiniLM-L6-v2`: Rápido, buena calidad (default)
  - `all-mpnet-base-v2`: Mejor calidad, más lento
- `device`: `cpu` o `cuda` (para GPU)
- `distance_function`:
  - `l2`: Distancia euclideana. El valor default
  - `cosine`: Similitud coseno
  - `ip`: Producto escalar

#### Crear Tu Propia Configuración

1. Copia una configuración existente:
```bash
cp configs/default.json configs/mi_config.json
```

2. Modifica los parámetros según tus necesidades

3. Úsala con el flag `--config`:
```bash
python main.py -z code.zip -p "pregunta" --config configs/mi_config.json
```

#### Uso Programático con Configs

```python
from src.config_loader import RAGConfig
from src.inserter import ChromaCollection

# Cargar configuración
config = RAGConfig.from_json("configs/optimal.json")

# Crear colección con la config
collection = ChromaCollection("mi_proyecto", config=config)

# El RAG usará automáticamente todos los parámetros de la config
respuesta = collection.rag("¿Cómo funciona esto?")
```

#### Trade-offs de Configuración

| Parámetro | ⬆️ Aumentar | ⬇️ Disminuir |
|-----------|------------|-------------|
| **k_documents** | Más contexto, más costo | Más rápido, menos contexto |
| **temperature** | Más creativo, menos preciso | Más determinístico |
| **parser_threshold** | Fragmentos grandes | Fragmentos granulares |
| **max_tokens** | Respuestas largas, más caro | Respuestas breves |

### Proceso completo

Cuando ejecutas el comando, el sistema:
1. Crea una colección de ChromaDB llamada según `-c` (default: "codeRAG")
2. Extraer y parsear todos los archivos soportados del zip
3. Insertar fragmentos de código en la base de datos vectorial
4. Ejecutar una consulta de ejemplo (puede ser modificada en el script)

### Uso Programático

#### 1. Detección de Lenguaje

Identifica tipos de archivo antes de procesarlos:

```python
from src.utils.language_detector import detect_language, is_supported, filter_supported_files

# Detectar lenguaje desde la ruta del archivo
lenguaje = detect_language('src/main.py')  # Retorna: 'python'

# Verificar si un archivo es soportado
if is_supported('app.js'):
    print("¡Archivo soportado!")

# Filtrar una lista de archivos
archivos = ['main.py', 'imagen.png', 'app.js', 'README.md']
archivos_codigo = filter_supported_files(archivos)
# Retorna: ['main.py', 'app.js', 'README.md']

# Obtener todas las extensiones soportadas
extensiones = get_supported_extensions()  # ['.py', '.js', '.md', ...]
```

#### 2. Parseo y Carga de Archivos

Parsea archivos individuales o archivos zip completos:

```python
from src.text_splitter import parse_file, load_from_zipfile

# Parsear un archivo individual
documentos = parse_file('src/main.py', parser_threshold=3)

# Cargar codebase completo desde zip
todos_los_docs = load_from_zipfile('codebase.zip')
```

#### 3. Trabajando con ChromaDB

Almacena y recupera fragmentos de código:

```python
from src.inserter import ChromaCollection
from langchain_core.documents import Document

# Inicializar una colección
coleccion = ChromaCollection('mi_proyecto')

# Insertar documentos
docs = [
    Document(page_content="def hola(): print('Hola')", metadata={"file": "main.py"}),
    Document(page_content="class Usuario: pass", metadata={"file": "models.py"})
]
coleccion.insert_docs(docs)

# Recuperar documentos similares
consulta = "cómo definir una función"
docs_similares, resultados = coleccion.retrieve_k_similar_docs([consulta], k=5)
print(docs_similares)
```

#### 4. Consultas RAG

Haz preguntas sobre tu codebase:

```python
from src.inserter import ChromaCollection

# Inicializar colección
coleccion = ChromaCollection('mi_proyecto')

# Consultar la codebase

respuesta = coleccion.rag(pregunta, model="gpt-4o-mini")
print(respuesta)

# Ejemplos de preguntas avanzadas:
# - "¿Qué tests le faltan a la función de login?"
# - "¿Cuál es el code coverage del proyecto?"
# - "Explícame la relación entre el archivo auth.py y database.py"
# - "¿Qué mejoras de seguridad recomiendas para este código?"
```

### Lenguajes y Extensiones Soportados

El sistema soporta las siguientes extensiones de archivo:

**Lenguajes de Programación:**
- Python: `.py`, `.pyw`, `.pyx`
- JavaScript/TypeScript: `.js`, `.mjs`, `.cjs`, `.jsx`, `.ts`, `.tsx`
- Java: `.java`
- C/C++: `.c`, `.h`, `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx`
- Go: `.go`
- Rust: `.rs`
- Ruby: `.rb`
- PHP: `.php`
- C#: `.cs`
- Swift: `.swift`
- Kotlin: `.kt`, `.kts`
- Scala: `.scala`
- Haskell: `.hs`
- Lua: `.lua`
- LaTeX: `.tex`

**Documentación y Datos:**
- Markdown: `.md`, `.markdown`
- JSON: `.json`
- YAML: `.yaml`, `.yml`
- TOML: `.toml`
- XML: `.xml`
- Texto: `.txt`, `.rst`
- Config: `.ini`, `.cfg`, `.conf`

**Web:**
- HTML: `.html`, `.htm`
- CSS: `.css`, `.scss`, `.sass`, `.less`

**Otros:**
- Shell scripts: `.sh`, `.bash`, `.zsh`
- SQL: `.sql`
- R: `.r`, `.R`

## Configuración Avanzada

### Parser Threshold

El parámetro `parser_threshold` (por defecto: 3) controla el tamaño mínimo de los fragmentos de código. Valores más bajos crean fragmentos más pequeños y granulares.

```python
documentos = parse_file('archivo.py', parser_threshold=5)  # Fragmentos más grandes
```

### Modelo de Embeddings

El sistema usa SentenceTransformer para generar embeddings (basado en BERT). Para cambiar el modelo, modifica la línea 11 en [src/inserter.py](src/inserter.py:11):

```python
_embedding_function = SentenceTransformerEmbeddingFunction(model_name="nombre-de-tu-modelo")
```

### Modelo GPT

El modelo por defecto es `gpt-4o-mini`. Puedes especificar un modelo diferente al llamar al método `rag()`:

```python
respuesta = coleccion.rag(consulta, model="gpt-4")
```

## Estructura del Proyecto

```
codebase-rag/
├── src/
│   ├── inserter.py              # Gestión de ChromaDB e implementación RAG
│   ├── text_splitter.py         # Parseo de archivos y text splitting
│   ├── config_loader.py         # Cargador de configuraciones JSON
│   ├── prompt_loader.py         # Cargador de plantillas de prompts
│   └── utils/
│       ├── __init__.py
│       └── language_detector.py # Utilidades de detección de lenguaje
├── configs/                      # Archivos de configuración JSON
│   ├── default.json
│   ├── optimal.json
│   ├── fast.json
│   └── detailed.json
├── prompts/                      # Plantillas de prompts
│   ├── default.txt
│   ├── optimal.txt
│   ├── detailed.txt
│   ├── concise.txt
│   ├── spanish.txt
│   └── beginner_friendly.txt
├── dataset/                      # Datasets de prueba
│   ├── jam-py-v7-develop.zip    # Proyecto web (JS/Python) ~15MB
│   ├── numpy-main.zip           # Librería científica (Python/C) ~10MB
│   └── SMTP-Protos-main.zip     # Protocolo de red (C) ~200KB
├── tests/
│   └── test_language_detector.py
├── context/                      # Material de referencia del curso
│   ├── propuesta.txt
│   ├── DeepLearningTP2.pdf
│   └── [notebooks de referencia]
├── main.py                       # Script principal CLI
├── requirements.txt
├── .env                          # Tus API keys (crear este archivo)
├── .gitignore
└── README.md
```

## Datasets de Prueba

El directorio `dataset/` contiene varios proyectos de código de diferentes tamaños y lenguajes de programación para probar y evaluar el sistema RAG:

### Proyectos Incluidos

1. **SMTP-Protos (main)** - ~200KB
   - Implementación de protocolo de red
   - Lenguaje: C
   - Tamaño: Pequeño
   - Ideal para: Pruebas rápidas, código de bajo nivel, protocolos livianos

2. **jam-py (v7-develop)** - ~15MB
   - Framework web full-stack
   - Lenguajes: JavaScript, Python
   - Tamaño: Pequeño-Mediano (archivos grandes)
   - Ideal para: Probar parseo multi-lenguaje, arquitecturas web

3. **NumPy (main)** - ~10MB
   - Librería científica de Python
   - Lenguajes: Python, C
   - Tamaño: Mediano-Grande
   - Ideal para: Integración Python/C, documentación técnica, código científico complejo

### Uso de los Datasets

Estos proyectos permiten evaluar el sistema en diferentes escenarios:
- **Variedad de lenguajes**: C, JavaScript, Python
- **Diferentes tamaños**: Desde pequeño (~200KB) hasta mediano-grande (~10MB)
- **Diversos dominios**: Protocolos de red, aplicaciones web, computación científica
- **Complejidad variada**: Protocolos simples vs librerías científicas complejas

**Ejemplo de uso:**
```bash
# Proyecto pequeño (C)
python main.py -z dataset/SMTP-Protos-main.zip -p "¿Cómo funciona el protocolo?"

# Proyecto pequeño-mediano (JS/Python)
python main.py -z dataset/jam-py-v7-develop.zip -p "Explicame la arquitectura del framework"

# Proyecto mediano-grande (Python/C)
python main.py -z dataset/numpy-main.zip -p "¿Qué hace la función array?"
```

## Desafíos Técnicos

Los principales desafíos técnicos abordados en este proyecto incluyen:

1. **Parseo de Código Multi-lenguaje**: Implementación de detección automática y parseo específico por lenguaje usando tree-sitter
2. **Fragmentación Inteligente**: División del contenido respetando la estructura y sintaxis de cada lenguaje
3. **Recuperación y Rankeo Eficiente**: Uso de embeddings semánticos para encontrar los fragmentos más relevantes
4. **Referenciado a las Fuentes**: Mantener metadata precisa de archivos y ubicaciones para cada fragmento

## Cómo Funciona

1. **Procesamiento de Archivos**: El sistema identifica archivos soportados usando el detector de lenguaje
2. **Parseo**: Los parsers específicos por lenguaje extraen segmentos de código significativos
3. **Embedding**: Los fragmentos de código se convierten a vectores usando sentence transformers
4. **Almacenamiento**: Los embeddings y metadata se almacenan en ChromaDB
5. **Retrieval**: Cuando haces una pregunta, el sistema encuentra los fragmentos más semánticamente similares
6. **Generación**: Los fragmentos relevantes se pasan a GPT junto con tu pregunta para generar una respuesta contextual

## Testing

Ejecuta los tests del detector de lenguaje:

```bash
python tests/test_language_detector.py
```

Este script prueba todas las funcionalidades del módulo de detección de lenguaje, incluyendo:
- Detección de lenguaje por extensión
- Validación de archivos soportados
- Filtrado de listas de archivos
- Obtención de extensiones y lenguajes soportados

## Dependencias Principales

- `chromadb`: Base de datos vectorial para almacenamiento de documentos
- `openai`: Integración con modelos GPT
- `langchain` y `langchain-community`: Procesamiento de documentos y text splitting
- `sentence-transformers`: Generación de embeddings
- `tree-sitter` y `tree-sitter-languages`: Parseo consciente del lenguaje para código
- `python-dotenv`: Gestión de variables de entorno

Ver [requirements.txt](requirements.txt) para la lista completa.

## Limitaciones Actuales

- Requiere OpenAI O Gemini API key (implica costos por uso de API)
- Los archivos binarios e imágenes no son soportados
- El rendimiento depende del tamaño de la codebase y el tamaño de fragmento elegido
- La calidad de las respuestas depende de la relevancia de los fragmentos recuperados

## Evaluación con RAGAS

El sistema incluye un módulo completo de evaluación basado en [RAGAS](https://docs.ragas.io/) (Retrieval-Augmented Generation Assessment) para medir la calidad del sistema RAG.

### Métricas Disponibles

El sistema soporta las siguientes métricas de RAGAS:

- **Faithfulness**: Evalúa si la respuesta generada se basa fielmente en el contexto recuperado (sin alucinaciones)
- **Answer Relevancy**: Mide qué tan relevante es la respuesta para la pregunta del usuario
- **Context Precision**: Evalúa si el contexto recuperado es preciso y relevante
- **Context Recall**: Mide qué tan completo es el contexto recuperado comparado con la respuesta ideal

### Métricas que Requieren Ground Truth (Referencias)

**IMPORTANTE**: Algunas métricas requieren respuestas de referencia (ground truth):
- `context_precision` ✓ requiere referencias
- `context_recall` ✓ requiere referencias
- `faithfulness` ✗ NO requiere referencias
- `answer_relevancy` ✗ NO requiere referencias

### Evaluación Rápida

Evaluar tu sistema RAG en 3 pasos:

```bash
# 1. Preparar dataset de prueba (ver data/evaluation/test_queries.json)
# 2. Ejecutar evaluación
python evaluate.py \
  --collection-name codeRAG \
  --test-dataset data/evaluation/test_queries.json \
  --metrics faithfulness,answer_relevancy

# 3. Ver resultados en results/evaluation_results.csv
```

### Uso Completo del Script de Evaluación

**Evaluar con dataset existente:**
```bash
python evaluate.py \
  --collection-name codeRAG \
  --test-dataset data/evaluation/test_queries.json \
  --eval-config configs/evaluation.json \
  --output results/eval_2025_01_18.csv \
  --verbose
```

**Auto-generar queries de prueba:**
```bash
python evaluate.py \
  --collection-name codeRAG \
  --auto-generate 20 \
  --metrics faithfulness,answer_relevancy \
  --save-dataset data/evaluation/my_test_queries.json
```

**Generar referencias automáticamente:**
```bash
python evaluate.py \
  --collection-name codeRAG \
  --test-dataset data/evaluation/code_specific_queries.json \
  --generate-references \
  --save-dataset data/evaluation/queries_with_refs.json
```

⚠️ **Nota**: Las referencias auto-generadas deben ser revisadas manualmente para garantizar calidad.

**Curacióninteractiva de referencias:**
```bash
python evaluate.py \
  --collection-name codeRAG \
  --test-dataset data/evaluation/test_queries.json \
  --interactive-curation
```

### Uso Programático

```python
from src.evaluation import RAGASEvaluator, EvaluationConfig, TestDatasetGenerator
from src.inserter import ChromaCollection

# 1. Cargar configuración de evaluación
eval_config = EvaluationConfig.from_json("configs/evaluation.json")

# 2. Inicializar evaluador (LLM aislado del sistema de generación)
evaluator = RAGASEvaluator(eval_config)

# 3. Preparar queries de prueba
generator = TestDatasetGenerator()
test_queries = generator.load_from_json("data/evaluation/test_queries.json")

# 4. Recolectar datos de evaluación
collection = ChromaCollection("codeRAG")
queries = [q.question for q in test_queries]
references = [q.reference for q in test_queries]

eval_data = collection.collect_evaluation_data(
    queries=queries,
    references=references,
    verbose=True
)

# 5. Ejecutar evaluación
from ragas import EvaluationDataset
dataset = EvaluationDataset.from_list(eval_data)

results = evaluator.evaluate(
    dataset=dataset,
    metrics=["faithfulness", "answer_relevancy"],
    verbose=True
)

# 6. Exportar resultados con versionado
evaluator.export_results(
    results=results,
    output_path="results/my_evaluation.csv",
    format="csv"
)
```

### Estructura de un Dataset de Evaluación

**Formato JSON:**
```json
{
  "samples": [
    {
      "question": "¿Cómo funciona la autenticación?",
      "reference": "La autenticación se maneja mediante...",
      "metadata": {
        "category": "security",
        "difficulty": "medium"
      }
    }
  ]
}
```

**Campos:**
- `question` (requerido): La pregunta a evaluar
- `reference` (opcional): Respuesta de referencia (ground truth)
- `metadata` (opcional): Metadata adicional para análisis

### Configuración de Evaluación

El archivo `configs/evaluation.json` permite configurar:

```json
{
  "evaluator_llm": {
    "provider": "google",
    "model": "gemini-1.5-flash",
    "temperature": 0.0
  },
  "metrics": ["faithfulness", "answer_relevancy"],
  "batch_size": 10,
  "warn_on_missing_fields": true
}
```

**Parámetros clave:**
- `evaluator_llm`: Modelo LLM **aislado** para evaluación (diferente del modelo de generación)
- `metrics`: Métricas a calcular
- `warn_on_missing_fields`: Advertir sobre campos faltantes según requerimientos de métricas

### Interpretando Resultados

**Rangos de puntuación (0.0 - 1.0):**
- **0.8 - 1.0**: Excelente - El sistema está funcionando muy bien
- **0.6 - 0.8**: Bueno - Rendimiento aceptable, posibles mejoras
- **0.4 - 0.6**: Regular - Necesita optimización
- **< 0.4**: Pobre - Requiere cambios significativos

**Métricas específicas para código:**
- **Faithfulness alto**: Respuestas basadas en código real (sin inventar funciones)
- **Answer Relevancy alto**: Respuestas centradas en la pregunta
- **Context Precision alto**: Fragmentos de código recuperados son relevantes
- **Context Recall alto**: Se recuperan todos los fragmentos necesarios

### Generación de Referencias (Ground Truth)

**Opción 1: Auto-generar con RAG**
```python
generator = TestDatasetGenerator()
queries = generator.load_from_json("queries_without_refs.json")

# Genera referencias candidatas
queries_with_refs = generator.build_reference_answers(
    queries=queries,
    collection=collection,
    verbose=True
)

# ⚠️ REVISAR MANUALMENTE las referencias generadas
generator.save_to_json(queries_with_refs, "queries_for_review.json")
```

**Opción 2: Curación interactiva**
```python
generator.curate_references(
    dataset_path="data/evaluation/test_queries.json",
    resume=True  # Saltar queries que ya tienen referencia
)
```

**Opción 3: Creación manual**
Editar directamente el archivo JSON con tus respuestas ideales.

### Versionado y Reproducibilidad

Los resultados exportados incluyen metadata completa para reproducibilidad:
- Versión de RAGAS, Python, y dependencias
- Modelo de generación y evaluación usados
- Modelo de embeddings
- Timestamp de evaluación
- Snapshot de configuración

Ver archivo `.metadata.json` junto al CSV de resultados.

### Buenas Prácticas

1. **Datasets balanceados**: Incluir preguntas de diferentes niveles de dificultad y categorías
2. **Referencias de calidad**: Para métricas que requieren ground truth, asegurar referencias precisas
3. **Evaluación periódica**: Evaluar después de cambios en configuración, modelos, o codebase
4. **Comparación de configs**: Usar evaluación para comparar diferentes configuraciones (optimal vs fast)
5. **Análisis de errores**: Revisar muestras con puntuación baja para identificar problemas

### Archivos de Evaluación

```
codebase-rag/
├── configs/
│   └── evaluation.json          # Config de evaluación
├── data/
│   └── evaluation/
│       ├── test_queries.json            # Queries con referencias
│       └── code_specific_queries.json   # Queries sin referencias
├── results/                     # Resultados de evaluación
│   ├── evaluation_results.csv
│   └── evaluation_results.metadata.json
├── src/
│   └── evaluation/
│       ├── ragas_evaluator.py           # Evaluador principal
│       └── test_dataset_generator.py    # Generador de datasets
├── tests/
│   └── test_ragas_evaluator.py
└── evaluate.py                  # Script CLI de evaluación
```

## Mejoras Futuras

- Soporte para modelos de embedding adicionales (opciones locales/open-source)
- Interfaz web para consultas más fáciles
- Soporte para actualizaciones incrementales de colecciones existentes
- Filtrado avanzado y búsqueda por metadata
- Evaluación de rendimiento entre repos con múltiples lenguajes vs un solo lenguaje
- Análisis de code coverage y generación de tests
- Soporte multi-modal para diagramas e imágenes de documentación
- **Métricas personalizadas de código**: Validación de sintaxis, correctitud de APIs, alineación código-documentación



### Objetivos Cubiertos

- ✅ Implementación de arquitectura RAG completa
- ✅ Integración de Data Loading, Text Splitting y Base de Datos Vectorial
- ✅ Técnicas de preprocesamiento (detección de lenguaje, filtrado, parseo)
- ✅ Dataset de código fuente en múltiples lenguajes
- ✅ Sistema de retrieval con embeddings semánticos

## Licencia

Este proyecto se proporciona tal cual para fines educativos y de desarrollo.
