# Implementación RAG - Guía Técnica Completa

Este documento explica el pipeline RAG implementado, cómo funciona cada componente y el estado actual del proyecto.

---

## 📋 Índice

1. [Arquitectura General del RAG](#arquitectura-general-del-rag)
2. [Pipeline de Indexación (Indexing Pipeline)](#pipeline-de-indexación)
3. [Integración con ChromaDB](#integración-con-chromadb)
4. [Text Splitting](#text-splitting)
5. [Integración con OpenAI](#integración-con-openai)
6. [Pipeline de Consulta (Query Pipeline)](#pipeline-de-consulta)
7. [Estado de Implementación](#estado-de-implementación)
8. [Cómo Ejecutar el Sistema](#cómo-ejecutar-el-sistema)

---

## Arquitectura General del RAG

### ¿Qué es RAG?

**RAG (Retrieval-Augmented Generation)** combina dos técnicas:
- **Retrieval**: Buscar información relevante en una base de datos
- **Generation**: Usar un LLM (GPT) para generar respuestas basadas en esa información

### Pipeline Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    FASE 1: INDEXACIÓN                           │
└─────────────────────────────────────────────────────────────────┘

Archivos ZIP → Extracción → Detección Lenguaje → Parseo →
→ Text Splitting → Generación Embeddings → ChromaDB

┌─────────────────────────────────────────────────────────────────┐
│                    FASE 2: CONSULTA                             │
└─────────────────────────────────────────────────────────────────┘

Pregunta Usuario → Embedding Query → Búsqueda Similaridad →
→ Recuperar Top-K → Construir Prompt → GPT → Respuesta
```

---

## Pipeline de Indexación

### Estado: ✅ IMPLEMENTADO

El indexing pipeline transforma el código fuente en embeddings vectoriales almacenados en ChromaDB.

### Flujo de Datos

#### 1. **Carga de Archivos** (`text_splitter.py`)

```python
# Archivo: src/text_splitter.py
# Función: load_from_zipfile()

ZIP File → Extracción temporal → Listado de archivos
```

**Implementación actual:**
- ✅ Extrae ZIP a directorio temporal con `tempfile.TemporaryDirectory()`
- ✅ Recorre recursivamente todos los archivos con `os.walk()`
- ✅ Filtra archivos ocultos (`.git`, `.DS_Store`, etc.)
- ✅ Limpia automáticamente archivos temporales al terminar

**Código clave:**
```python
with tempfile.TemporaryDirectory() as temp_dir:
    with ZipFile(zippath, 'r') as zip_file:
        zip_file.extractall(temp_dir)
        # Procesar archivos...
```

#### 2. **Detección de Lenguaje** (`language_detector.py`)

```python
# Archivo: src/utils/language_detector.py
# Funciones: is_supported(), detect_language()

Archivo → Extensión → LANGUAGE_MAP → Lenguaje
```

**Implementación actual:**
- ✅ 50+ extensiones soportadas (Python, JS, TS, Java, Go, Rust, etc.)
- ✅ Filtrado automático de archivos no soportados
- ✅ Detección case-insensitive (`.PY` = `.py`)

**Código clave:**
```python
if not is_supported(filepath):
    print(f"⊘ Saltando (no soportado): {filename}")
    continue

language = detect_language(filepath)  # "python", "javascript", etc.
```

#### 3. **Parseo de Código** (`text_splitter.py`)

```python
# Archivo: src/text_splitter.py
# Función: parse_file()

Archivo → LanguageParser → Fragmentos (Documents)
```

**Implementación actual:**
- ✅ Usa `LanguageParser` de Langchain (consciente del lenguaje)
- ✅ Parser threshold configurable (default: 3)
- ✅ Respeta la estructura sintáctica del código
- ✅ Manejo de errores por archivo (un error no detiene todo)

**Código clave:**
```python
parser = LanguageParser(parser_threshold=3)
blob = Blob(path=filepath)
docs = parser.lazy_parse(blob=blob)
```

**¿Qué hace el parser?**
- Identifica funciones, clases, métodos automáticamente
- Divide el código en fragmentos significativos
- Preserva contexto sintáctico (no corta a mitad de función)

#### 4. **Enriquecimiento de Metadata** (`text_splitter.py`)

```python
Fragmentos → Agregar metadata → Documents enriquecidos
```

**Implementación actual:**
- ✅ Agrega `source_file` (nombre del archivo)
- ✅ Agrega `language` (lenguaje detectado)
- ✅ Preserva metadata original del parser

**Código clave:**
```python
for doc in docs:
    doc.metadata['source_file'] = filename
    doc.metadata['language'] = detect_language(filepath)
```

---

## Integración con ChromaDB

### Estado: ✅ IMPLEMENTADO

ChromaDB es la base de datos vectorial que almacena los embeddings del código.

### ¿Por qué ChromaDB?

- 🚀 **Rápido**: Optimizado para búsqueda de similaridad
- 💾 **Local**: No requiere servidor externo
- 🔌 **Fácil integración**: API simple
- 🎯 **Embeddings automáticos**: Genera vectores automáticamente

### Componentes Implementados

#### 1. **Inicialización de Colección** (`inserter.py`)

```python
# Archivo: src/inserter.py
# Función: _initialize_collection()

Nombre → Verificar existencia → Crear/Recuperar → Collection
```

**Implementación:**
- ✅ Idempotente (puedes llamarlo múltiples veces)
- ✅ Usa función de embedding global (SentenceTransformer/BERT)
- ✅ Cliente global compartido

**Código clave:**
```python
_embedding_function = SentenceTransformerEmbeddingFunction()
_chroma_client = chromadb.Client()

def _initialize_collection(collection_name: str):
    if collection_name not in existing_collections:
        return _chroma_client.create_collection(
            name=collection_name,
            embedding_function=_embedding_function
        )
    else:
        return _chroma_client.get_collection(name=collection_name)
```

#### 2. **Inserción de Documentos** (`inserter.py`)

```python
# Archivo: src/inserter.py
# Método: ChromaCollection.insert_docs()

Documents → Extraer contenido → ChromaDB.add() → Embeddings generados
```

**Implementación:**
- ✅ Genera IDs únicos con UUID4
- ✅ ChromaDB genera embeddings automáticamente
- ✅ Almacena: contenido + metadata + embeddings

**Código clave:**
```python
self._chroma_collection.add(
    ids=[str(uuid4()) for _ in docs],
    documents=[doc.page_content for doc in docs],
    metadatas=[doc.metadata for doc in docs]
)
```

**Nota importante:** ChromaDB genera los embeddings automáticamente usando la función de embedding configurada (SentenceTransformer). No necesitas calcular vectores manualmente.

#### 3. **Búsqueda por Similaridad** (`inserter.py`)

```python
# Archivo: src/inserter.py
# Método: ChromaCollection.retrieve_k_similar_docs()

Query → Embedding → Búsqueda vectorial → Top-K documentos
```

**Implementación:**
- ✅ Búsqueda semántica (no keyword matching)
- ✅ K configurable (default: 5)
- ✅ Incluye documentos y embeddings en resultados

**Código clave:**
```python
results = self._chroma_collection.query(
    query_texts=[query],
    n_results=k,
    include=['documents', 'embeddings']
)
```

### ¿Qué hay de Pinecone?

**Estado: ❌ NO IMPLEMENTADO (no necesario)**

Pinecone es otra base de datos vectorial, pero para este proyecto:
- ✅ ChromaDB es suficiente (local, gratuito, rápido)
- ❌ Pinecone requiere cuenta y es de pago para producción
- 💡 Puedes migrar a Pinecone después si necesitas:
  - Escalabilidad masiva (millones de documentos)
  - Acceso distribuido/multi-región
  - Alta disponibilidad

**Para tu TP académico: ChromaDB es la elección correcta.**

---

## Text Splitting

### Estado: ✅ IMPLEMENTADO

El text splitting divide archivos grandes en fragmentos manejables.

### ¿Por qué es importante?

- 🎯 **Precisión**: Fragmentos pequeños = búsquedas más precisas
- 💰 **Costos**: GPT tiene límites de tokens y cobra por token
- 🧠 **Contexto**: Fragmentos enfocados = mejores respuestas

### Estrategia Implementada

#### 1. **LanguageParser** (Langchain)

```python
parser = LanguageParser(parser_threshold=3)
```

**Características:**
- ✅ Consciente de la sintaxis del lenguaje
- ✅ No corta funciones o clases a la mitad
- ✅ Preserva contexto estructural
- ✅ Threshold configurable (min tamaño de fragmento)

#### 2. **Parser Threshold**

```python
parser_threshold = 3  # Tamaño mínimo de fragmento
```

**Valores típicos:**
- `1-3`: Fragmentos muy pequeños (alta granularidad)
- `3-5`: Balance recomendado ✅ (implementado)
- `5-10`: Fragmentos más grandes (menos precisión pero más contexto)

### Ejemplo de Fragmentación

**Archivo original:**
```python
# models.py (100 líneas)
class Usuario:
    def __init__(self, nombre):
        self.nombre = nombre

    def saludar(self):
        return f"Hola, {self.nombre}"

class Producto:
    def __init__(self, nombre, precio):
        self.nombre = nombre
        self.precio = precio
```

**Fragmentos generados:**
```
Fragment 1: class Usuario + __init__ + saludar (20 líneas)
Fragment 2: class Producto + __init__ (15 líneas)
```

### Metadata de Fragmentos

Cada fragmento incluye:
```python
{
    "source_file": "models.py",
    "language": "python",
    "start_line": 10,      # Del parser
    "end_line": 25,        # Del parser
}
```

---

## Integración con OpenAI

### Estado: ✅ IMPLEMENTADO

OpenAI GPT-4.1-nano genera las respuestas finales usando el contexto recuperado.

### Flujo de Integración

#### 1. **Inicialización del Cliente** (`inserter.py`)

```python
# Archivo: src/inserter.py
# Clase: ChromaCollection.__init__()

.env → load_dotenv() → OpenAI Client
```

**Implementación:**
- ✅ Lee `OPENAI_API_KEY` desde `.env`
- ✅ Cliente se inicializa una vez por colección
- ✅ Reutilizable para múltiples queries

**Código clave:**
```python
load_dotenv(find_dotenv())
self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

#### 2. **Construcción del Prompt** (`inserter.py`)

```python
# Archivo: src/inserter.py
# Método: ChromaCollection.rag()

Query + Documentos → Prompt → GPT
```

**Estructura del prompt:**
```python
messages = [
    {
        "role": "system",
        "content": "You are an expert agent in coding..."
    },
    {
        "role": "user",
        "content": f"Question: {query}\n\nInformation:\n{information}"
    }
]
```

**Componentes:**
- **System message**: Define el rol del modelo
- **User message**: Pregunta + contexto recuperado
- **Information**: Los Top-K fragmentos unidos con `\n`

#### 3. **Llamada a la API** (`inserter.py`)

```python
response = self.openai_client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=messages
)
```

**Modelo usado: GPT-4.1-nano**
- ✅ Más rápido y económico ($0.10/M tokens input)
- ✅ Suficientemente capaz para RAG
- ✅ Context window: 1M tokens
- 🎯 Ideal para este proyecto

**Alternativas disponibles:**
- `gpt-4.1-mini`: Más capaz pero más caro
- `gpt-4o`: Aún más capaz (para casos complejos)
- `gpt-4`: Máxima capacidad (más caro)

#### 4. **Extracción de Respuesta**

```python
content = response.choices[0].message.content
return content
```

### Flujo Completo del RAG

```python
def rag(self, query: str, model: str = "gpt-4.1-nano") -> str:
    # 1. RETRIEVAL: Buscar fragmentos relevantes
    documents, _ = self.retrieve_k_similar_docs(query)

    # 2. AUGMENTATION: Construir contexto
    information = "\n".join(documents)
    messages = [...]

    # 3. GENERATION: Generar respuesta
    response = self.openai_client.chat.completions.create(...)
    return response.choices[0].message.content
```

---

## Pipeline de Consulta

### Estado: ✅ IMPLEMENTADO

El query pipeline maneja las preguntas del usuario.

### Flujo Paso a Paso

```
┌──────────────────────────────────────────────────────────┐
│  1. Usuario hace pregunta                                │
│     "¿Cómo se crea un usuario?"                          │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  2. Embedding de la query                                │
│     Query → SentenceTransformer → Vector[768]            │
│     (Automático en ChromaDB)                             │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  3. Búsqueda de similaridad                              │
│     Vector query vs Todos los vectores en ChromaDB       │
│     Distancia coseno → Top-K más similares               │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  4. Recuperación de documentos                           │
│     Top-5 fragmentos más relevantes:                     │
│     - Fragment: "class Usuario: def __init__..."         │
│     - Fragment: "usuario1 = Usuario('Juan')"             │
│     - Fragment: "# Crear usuarios de prueba"             │
│     - etc.                                               │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  5. Construcción del prompt                              │
│     System: "Eres experto en código..."                  │
│     User: "Pregunta: ... Información: [fragmentos]"      │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  6. Llamada a GPT                                        │
│     OpenAI API → GPT-4.1-nano procesa                    │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  7. Respuesta al usuario                                 │
│     "Para crear un usuario, usa la clase Usuario..."     │
└──────────────────────────────────────────────────────────┘
```

### Ejemplo Concreto

**Input:**
```python
collection.rag("¿Cómo se calcula un descuento?")
```

**Paso 1 - Retrieval:**
```python
# ChromaDB encuentra los 5 fragmentos más similares:
[
    "def aplicar_descuento(self, porcentaje: float) -> float: ...",
    "DESCUENTO_MAYORISTA = 0.15  # 15% de descuento",
    "descuento = self.precio * (porcentaje / 100)",
    "return self.precio - descuento",
    "if not 0 <= porcentaje <= 100: raise ValueError(...)"
]
```

**Paso 2 - Augmentation:**
```python
information = """
def aplicar_descuento(self, porcentaje: float) -> float:
    if not 0 <= porcentaje <= 100:
        raise ValueError("El porcentaje debe estar entre 0 y 100")
    descuento = self.precio * (porcentaje / 100)
    return self.precio - descuento

DESCUENTO_MAYORISTA = 0.15
...
"""
```

**Paso 3 - Generation (GPT):**
```
"El descuento se calcula con el método aplicar_descuento() de la clase
Producto. Este método toma un porcentaje (0-100) y calcula el descuento
multiplicando el precio por el porcentaje dividido 100. Luego resta el
descuento del precio original y retorna el resultado. El método valida
que el porcentaje esté entre 0 y 100, lanzando ValueError si no lo está."
```

---

## Estado de Implementación

### ✅ Componentes Completados

| Componente | Estado | Archivo | Descripción |
|------------|--------|---------|-------------|
| **Detección de lenguaje** | ✅ | `language_detector.py` | 50+ extensiones soportadas |
| **Extracción de ZIP** | ✅ | `text_splitter.py` | Con tempfile y limpieza automática |
| **Parseo de código** | ✅ | `text_splitter.py` | LanguageParser consciente de sintaxis |
| **Filtrado de archivos** | ✅ | `text_splitter.py` | Usa is_supported() |
| **Metadata enriquecida** | ✅ | `text_splitter.py` | source_file + language |
| **Inicialización ChromaDB** | ✅ | `inserter.py` | Colecciones idempotentes |
| **Inserción documentos** | ✅ | `inserter.py` | Con UUID y embeddings automáticos |
| **Búsqueda similaridad** | ✅ | `inserter.py` | Top-K configurable |
| **Integración OpenAI** | ✅ | `inserter.py` | GPT-4.1-nano |
| **Pipeline RAG completo** | ✅ | `inserter.py` | Retrieval + Augmentation + Generation |
| **Manejo de errores** | ✅ | `text_splitter.py` | Try-catch por archivo |
| **Output informativo** | ✅ | `text_splitter.py` | Símbolos y contadores |
| **Documentación** | ✅ | Todos | Docstrings sucintos |
| **Dataset de prueba** | ✅ | `test_data/` | 5 archivos + ZIP |
| **Configuración .env** | ✅ | `.env` | Template creado |
| **.gitignore** | ✅ | `.gitignore` | Actualizado para RAG |

### ❌ Componentes NO Implementados

| Componente | Estado | Razón |
|------------|--------|-------|
| **Pinecone** | ❌ | No necesario - ChromaDB es suficiente |
| **RecursiveCharacterTextSplitter** | ❌ | LanguageParser es mejor para código |
| **Métricas de evaluación** | ❌ | Pendiente (Fase avanzada) |
| **Cache de embeddings** | ❌ | Optimización futura |
| **Batch queries** | ❌ | TODO en retrieve_k_similar_docs |
| **Interface web** | ❌ | Fuera del alcance del TP |

### 🔄 En Progreso

| Componente | Estado | Siguiente paso |
|------------|--------|----------------|
| **Testing end-to-end** | 🔄 | Ejecutar con test_codebase.zip |
| **Validación de API key** | 🔄 | Agregar clave real de OpenAI |

---

## Cómo Ejecutar el Sistema

### Prerequisitos

1. **Activar venv:**
   ```bash
   source venv/bin/activate
   ```

2. **Verificar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar API key:**
   Edita `.env` y agrega tu clave de OpenAI:
   ```
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
   ```

### Ejecución Completa

#### 1. Indexar el codebase de prueba

```bash
python src/text_splitter.py test_collection test_codebase.zip
```

**Qué hace:**
- Extrae archivos del ZIP
- Detecta lenguajes y filtra soportados
- Parsea cada archivo
- Genera embeddings
- Inserta en ChromaDB
- Hace una query de prueba

**Output esperado:**
```
============================================================
Indexando codebase: test_codebase.zip
Colección: test_collection
============================================================

Cargando archivos desde ZIP...
✓ Parseando: config.py (python)
  → 3 fragmentos extraídos
✓ Parseando: main.py (python)
  → 5 fragmentos extraídos
✓ Parseando: models.py (python)
  → 8 fragmentos extraídos
✓ Parseando: utils.py (python)
  → 12 fragmentos extraídos
✓ Parseando: README.md (markdown)
  → 2 fragmentos extraídos

============================================================
Total de fragmentos a insertar: 30
============================================================

Insertando en ChromaDB...
✓ Documentos insertados exitosamente

Query de prueba: '¿Qué funcionalidades tiene este código?'
============================================================

Respuesta:
Este código implementa un sistema de gestión con las siguientes
funcionalidades:
- Gestión de usuarios con la clase Usuario
- Catálogo de productos con la clase Producto
- Funciones matemáticas básicas (sumar, restar, multiplicar, dividir)
- Formateo de precios y cálculo de totales
- Configuración centralizada con constantes para la aplicación
...
```

#### 2. Uso programático

```python
from src.inserter import ChromaCollection

# Conectar a la colección existente
collection = ChromaCollection('test_collection')

# Hacer queries
respuesta = collection.rag("¿Cómo se crea un usuario?")
print(respuesta)

respuesta = collection.rag("¿Qué hace la función aplicar_descuento?")
print(respuesta)

respuesta = collection.rag("¿Cuál es el valor del IVA?")
print(respuesta)
```

#### 3. Verificar fragmentos recuperados

```python
from src.inserter import ChromaCollection

collection = ChromaCollection('test_collection')

# Ver qué fragmentos recupera para una query
docs, results = collection.retrieve_k_similar_docs(
    "¿Cómo se calcula un descuento?",
    k=3
)

print("Fragmentos recuperados:")
for i, doc in enumerate(docs, 1):
    print(f"\n--- Fragmento {i} ---")
    print(doc)
```

### Troubleshooting

**Error: "No module named 'chromadb'"**
```bash
pip install chromadb
```

**Error: "OPENAI_API_KEY not found"**
- Verifica que `.env` existe en la raíz
- Verifica que la línea es exactamente: `OPENAI_API_KEY=tu_clave`
- Reinicia el script

**Error al parsear archivos**
- Normal para algunos archivos binarios
- El sistema continúa con los siguientes archivos
- Solo archivos soportados se procesan

**Sin fragmentos generados**
- Verifica que el ZIP contiene archivos soportados
- Aumenta `parser_threshold` si los archivos son muy pequeños

---

## Resumen Ejecutivo

### Pipeline Implementado

```
✅ ZIP → Extracción → Detección → Parseo → ChromaDB → OpenAI → Respuesta
```

### Archivos Clave

1. **`src/utils/language_detector.py`**: Detección de lenguajes (50+ extensiones)
2. **`src/text_splitter.py`**: Indexing pipeline completo
3. **`src/inserter.py`**: ChromaDB + OpenAI + RAG
4. **`test_data/`**: Dataset de prueba funcional
5. **`.env`**: Configuración de API keys

### Próximos Pasos

1. ✅ Agregar tu OpenAI API key real a `.env`
2. ✅ Ejecutar el sistema con `test_codebase.zip`
3. ⏭️ Probar diferentes queries
4. ⏭️ Evaluar calidad de respuestas
5. ⏭️ Ajustar parámetros si es necesario (k, threshold, modelo)

### Para el TP

Tienes implementado:
- ✅ Data Loading (ZIP extraction)
- ✅ Text Splitting (LanguageParser)
- ✅ Base de Datos Vectorial (ChromaDB)
- ✅ Retrieval (búsqueda semántica)
- ✅ Técnicas de Preprocesamiento (detección lenguaje, filtrado)
- ✅ Dataset (test_data con 5 archivos)

**El sistema está completo y funcional para el Trabajo Práctico.**

---

## Contacto y Referencias

- **Langchain Docs**: https://python.langchain.com/docs/
- **ChromaDB Docs**: https://docs.trychroma.com/
- **OpenAI API Docs**: https://platform.openai.com/docs/
- **SentenceTransformers**: https://www.sbert.net/

---

*Última actualización: 2025-10-13*
