# Prompts para RAG

Esta carpeta contiene templates de prompts del sistema para el RAG.

## Prompts Disponibles

### `default.txt`
Prompt estándar y balanceado. Útil para la mayoría de casos.

**Características:**
- Respuestas equilibradas en detalle
- Lenguaje técnico pero accesible
- Usa solo los fragmentos proporcionados

### `detailed.txt`
Para respuestas exhaustivas y completas.

**Características:**
- Explicaciones paso a paso
- Menciona todas las clases, funciones y variables
- Describe relaciones entre componentes
- Ideal para documentación o entender código complejo

### `concise.txt`
Para respuestas breves y directas.

**Características:**
- Respuestas cortas
- Solo información esencial
- Sin detalles innecesarios
- Ideal para consultas rápidas

### `spanish.txt`
Respuestas en español.

**Características:**
- Todo en español
- Terminología técnica en español
- Mantiene precisión técnica

### `beginner_friendly.txt`
Para explicaciones educativas.

**Características:**
- Lenguaje simple
- Evita jerga técnica o la explica
- Usa analogías y ejemplos
- Tono alentador
- Ideal para aprender

## Cómo Usar

### Desde código Python:

```python
from src.inserter import ChromaCollection

# Usar prompt por defecto
collection = ChromaCollection("mi_proyecto")

# Usar prompt específico
collection = ChromaCollection("mi_proyecto", prompt_template="detailed")
collection = ChromaCollection("mi_proyecto", prompt_template="spanish")
```

**Nota:** El prompt se define al crear la colección y no se puede cambiar después.

### Ejemplos de uso:

```python
from src.inserter import ChromaCollection
from src.text_splitter import load_from_zipfile

# Cargar documentos
docs = load_from_zipfile("mi_codigo.zip")

# Para documentación completa
collection = ChromaCollection("proyecto", prompt_template="detailed")
collection.insert_docs(docs)
respuesta = collection.rag("¿Cómo funciona la autenticación?")

# Para respuestas rápidas (crear nueva colección)
collection_concise = ChromaCollection("proyecto2", prompt_template="concise")
collection_concise.insert_docs(docs)
respuesta = collection_concise.rag("¿Qué hace main.py?")

# En español
collection_es = ChromaCollection("proyecto3", prompt_template="spanish")
collection_es.insert_docs(docs)
respuesta = collection_es.rag("¿Cuáles son las clases principales?")
```

## Crear Nuevos Prompts

Para crear un nuevo prompt:

1. Crea un archivo `.txt` en esta carpeta
2. El nombre del archivo será el nombre del prompt
3. Escribe el system prompt en el archivo

**Ejemplo:**

```bash
# Crear prompts/expert.txt
cat > prompts/expert.txt << 'EOF'
You are a senior software architect with 20 years of experience.
Provide expert-level analysis of code, including:
- Architecture patterns used
- Design decisions and trade-offs
- Performance implications
- Security considerations
- Best practices and anti-patterns

Use only the provided code snippets.
EOF
```

Luego úsalo:

```python
collection = ChromaCollection("proyecto", prompt_template="expert")
```

## Buenas Prácticas

1. **Sé específico**: Describe claramente el tono y nivel de detalle esperado
2. **Limita el scope**: Recuerda que el modelo solo ve los fragmentos recuperados
3. **Consistencia**: Mantén un formato similar entre prompts
4. **Testea**: Prueba cada prompt con diferentes tipos de preguntas

## Variables Futuras

(Actualmente no implementado, pero posible agregar)

Podrías usar variables como:
- `{query}`: La pregunta del usuario
- `{context}`: Los fragmentos de código
- `{language}`: Lenguaje detectado
- `{file}`: Archivo fuente

Ejemplo:
```
You are analyzing {language} code from {file}.
User question: {query}
Context: {context}
```
