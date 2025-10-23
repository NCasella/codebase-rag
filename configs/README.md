# Configuraciones RAG

Este directorio contiene archivos de configuración JSON para diferentes casos de uso del sistema RAG.

## Configuraciones Disponibles

### 1. `default.json` - Configuración Balanceada
**Uso:** General, balance entre calidad y costo

```bash
python main.py -z code.zip -p "pregunta" --config configs/default.json
```

**Características:**
- Modelo: `gpt-4.1-nano`
- K documentos: 5
- Temperature: 0.1
- Embeddings: `all-MiniLM-L6-v2`

---

### 2. `optimal.json` - Máxima Calidad ⭐ CON RERANKING
**Uso:** Análisis crítico, máxima precisión

```bash
python main.py -z code.zip -p "pregunta" --config configs/optimal.json
```

**Características:**
- Modelo: `gpt-5-mini` (mejor modelo)
- K documentos: 8 (después de reranking)
- Temperature: 1.0
- Embeddings: `all-mpnet-base-v2` (mejor calidad)
- Max context: 12000 caracteres
- **Reranking: Cross-Encoder habilitado** 🔥
  - Recupera 20 documentos, reranquea y selecciona los 8 mejores
  - Mejora significativa en relevancia de documentos

---

### 3. `fast.json` - Respuestas Rápidas
**Uso:** Consultas rápidas, desarrollo, experimentación

```bash
python main.py -z code.zip -p "pregunta" --config configs/fast.json
```

**Características:**
- Modelo: `gpt-3.5-turbo` (más rápido y barato)
- K documentos: 3
- Max tokens: 500 (respuestas cortas)
- Temperature: 0.2
- Max context: 4000 caracteres

---

### 4. `detailed.json` - Explicaciones Exhaustivas
**Uso:** Análisis profundo, documentación

```bash
python main.py -z code.zip -p "pregunta" --config configs/detailed.json
```

**Características:**
- Modelo: `gpt-4.1-nano`
- K documentos: 7
- Max tokens: 1500 (respuestas largas)
- Temperature: 0.15
- Prompt: `detailed` (explicaciones paso a paso)

---

### 5. `test_generator.json` - Generación de Tests ⭐ NUEVO
**Uso:** Generar código de tests unitarios y de integración

```bash
python main.py -z code.zip -p "genera tests para la función login en auth.py" --config configs/test_generator.json
```

**Características:**
- Modelo: `gpt-4o` (mejor comprensión de código)
- K documentos: 7 (más contexto para entender dependencias)
- Max tokens: 2000 (espacio para tests completos)
- Temperature: 0.2 (balance entre creatividad y precisión)
- Prompt: `test_generator` (especializado en testing)
- Parser threshold: 4 (fragmentos más grandes para mejor contexto)

**Ejemplos de uso:**

```bash
# Generar tests para una función específica
python main.py -z mi_proyecto.zip \
  -p "genera tests unitarios para la función calculate_total en utils.py" \
  --config configs/test_generator.json -v

# Generar tests para una clase
python main.py -z mi_proyecto.zip \
  -p "genera tests para la clase UserService" \
  --config configs/test_generator.json

# Generar tests de integración
python main.py -z mi_proyecto.zip \
  -p "genera tests de integración para el flujo de autenticación" \
  --config configs/test_generator.json

# Generar tests para edge cases
python main.py -z mi_proyecto.zip \
  -p "genera tests que cubran edge cases para la función parse_date" \
  --config configs/test_generator.json
```

**Qué incluye el test generado:**
- ✅ Imports necesarios del framework de testing
- ✅ Setup y fixtures si son necesarios
- ✅ Tests para happy path (casos normales)
- ✅ Tests para edge cases (límites, casos extremos)
- ✅ Tests para error handling (excepciones, validaciones)
- ✅ Mocks/stubs para dependencias externas
- ✅ Assertions claras y descriptivas
- ✅ Nombres de tests descriptivos
- ✅ Comentarios explicativos
- ✅ Comando para ejecutar los tests

**Frameworks soportados automáticamente:**
- Python: `pytest`, `unittest`
- JavaScript/TypeScript: `jest`, `mocha`, `vitest`
- Java: `JUnit`, `TestNG`
- Go: `testing`
- Rust: `#[test]`
- Y más...

---

### 6. `rerank_cross_encoder.json` - Reranking con Cross-Encoder ⭐ NUEVO
**Uso:** Máxima precisión en selección de documentos relevantes

```bash
python main.py -z code.zip -p "pregunta" --config configs/rerank_cross_encoder.json
```

**Características:**
- Modelo: `gemini-2.5-flash-lite`
- Reranking: **Cross-Encoder** (máxima precisión)
- Recupera: 20 documentos iniciales
- Selecciona: 5 mejores después de reranking
- Ideal para: Queries complejas donde la relevancia exacta es crítica

**¿Cuándo usar Cross-Encoder?**
- ✅ Queries técnicas específicas
- ✅ Búsqueda de bugs o vulnerabilidades
- ✅ Análisis de código crítico
- ⚠️ Más lento que retrieval normal (evalúa cada par query-documento)

---

### 7. `rerank_mmr.json` - Reranking con MMR (Diversidad) ⭐ NUEVO
**Uso:** Documentos diversos sin redundancia

```bash
python main.py -z code.zip -p "pregunta" --config configs/rerank_mmr.json
```

**Características:**
- Modelo: `gemini-2.5-flash-lite`
- Reranking: **MMR** (Maximal Marginal Relevance)
- Recupera: 15 documentos iniciales
- Selecciona: 5 diversos después de reranking
- Lambda: 0.7 (70% relevancia, 30% diversidad)
- Ideal para: Obtener diferentes perspectivas del código

**¿Cuándo usar MMR?**
- ✅ Explorar diferentes partes del código
- ✅ Evitar fragmentos repetitivos
- ✅ Obtener overview completo del sistema
- ⚠️ Rápido (no requiere modelo adicional)

---

## Crear Tu Propia Configuración

1. Copia una configuración base:
```bash
cp configs/default.json configs/mi_config.json
```

2. Edita los parámetros según tus necesidades

3. Úsala con el flag `--config`:
```bash
python main.py -z code.zip -p "pregunta" --config configs/mi_config.json
```

## Parámetros Configurables

### `prompt`
- `template`: Plantilla de prompt a usar
  - `default` - Balanceado
  - `optimal` - Análisis preciso
  - `detailed` - Explicaciones exhaustivas
  - `concise` - Respuestas breves
  - `spanish` - Respuestas en español
  - `beginner_friendly` - Tono educativo
  - `test_generator` - Generación de tests ⭐
- `max_context_length`: Límite de caracteres del contexto (4000-12000)

### `model`
- `name`: Modelo de OpenAI
  - `gpt-4o` - Mejor calidad (más caro)
  - `gpt-4.1-nano` - Balanceado
  - `gpt-3.5-turbo` - Rápido (más barato)
- `temperature`: 0.0 (determinístico) a 2.0 (creativo)
- `max_tokens`: Límite de tokens en respuesta (null = sin límite)
- `top_p`: Nucleus sampling (0.0-1.0)

### `retrieval`
- `k_documents`: Número de fragmentos a recuperar (3-10)
- `similarity_threshold`: Filtro de similitud mínima (0.0-1.0, null = sin filtro)

### `text_splitting`
- `parser_threshold`: Tamaño mínimo de fragmentos (2-5)
  - Bajo (2): Fragmentos pequeños, más granulares
  - Alto (5): Fragmentos grandes, más contexto

### `embeddings`
- `model_name`: Modelo de sentence-transformers
  - `all-MiniLM-L6-v2` - Rápido, buena calidad (default)
  - `all-mpnet-base-v2` - Mejor calidad, más lento
- `device`: `cpu` o `cuda` (para GPU)

### `rerank` ⭐ NUEVO
Mejora la relevancia de documentos recuperados mediante reranking.

- `enabled`: `true` para activar reranking, `false` para desactivar
- `strategy`: Estrategia de reranking
  - `none` - Sin reranking (default)
  - `cross_encoder` - Usa Cross-Encoder para scoring preciso (más lento, más preciso)
  - `mmr` - Maximal Marginal Relevance para diversidad (rápido, evita redundancia)
- `retrieve_k`: Número de documentos a recuperar antes de reranking (ej: 20)
- `top_n`: Número de documentos a seleccionar después de reranking (ej: 5)
- `cross_encoder_model`: Modelo de cross-encoder a usar (solo para strategy=cross_encoder)
  - `cross-encoder/ms-marco-MiniLM-L-12-v2` - Balanceado (default)
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` - Más rápido
- `cross_encoder_device`: `cpu` o `cuda` (para GPU)
- `mmr_lambda`: Balance relevancia/diversidad para MMR (0.0-1.0)
  - `0.0` - Máxima diversidad
  - `0.5` - Balanceado (default)
  - `1.0` - Máxima relevancia

## Trade-offs

| Parámetro | ⬆️ Aumentar | ⬇️ Disminuir |
|-----------|------------|-------------|
| **k_documents** | Más contexto, mejor comprensión<br>⚠️ Más costo, más lento | Respuestas rápidas<br>⚠️ Puede perder contexto |
| **temperature** | Más creativo/variado<br>⚠️ Menos preciso | Más determinístico<br>⚠️ Menos creativo |
| **max_tokens** | Respuestas completas<br>⚠️ Más caro | Respuestas concisas<br>⚠️ Puede truncar |
| **parser_threshold** | Fragmentos con más contexto<br>⚠️ Menos granularidad | Fragmentos granulares<br>⚠️ Puede perder contexto |
| **rerank.retrieve_k** ⭐ | Más candidatos para reranking<br>⚠️ Más lento (especialmente con cross-encoder) | Más rápido<br>⚠️ Puede perder buenos candidatos |
| **rerank.mmr_lambda** ⭐ | Más relevancia, menos diversidad | Más diversidad, menos relevancia |

## Casos de Uso por Configuración

| Tarea | Config Recomendada | Por qué |
|-------|-------------------|---------|
| Entender código nuevo | `optimal.json` | Máxima precisión + reranking |
| Debugging rápido | `fast.json` | Respuestas rápidas |
| Generar documentación | `detailed.json` | Explicaciones exhaustivas |
| Generar tests | `test_generator.json` ⭐ | Prompt especializado |
| Explicar a principiantes | `default.json` + prompt `beginner_friendly` | Balance + tono educativo |
| Análisis de seguridad | `rerank_cross_encoder.json` ⭐ | Máxima precisión en selección |
| Code review | `detailed.json` | Análisis profundo |
| Explorar arquitectura | `rerank_mmr.json` ⭐ | Diversidad de componentes |
| Búsqueda técnica precisa | `rerank_cross_encoder.json` ⭐ | Relevancia exacta |

## Ejemplos Completos

### Generar Tests Completos
```bash
# 1. Indexar proyecto
python main.py -z mi_api.zip \
  -p "genera tests completos para el endpoint POST /users en api/routes.py, incluyendo validación, errores y casos edge" \
  --config configs/test_generator.json \
  -v

# 2. Ver fragmentos recuperados (verbose) y código de test generado
```

### Análisis de Código Crítico
```bash
python main.py -z backend.zip \
  -p "analiza la función process_payment y lista todos los posibles bugs o vulnerabilidades" \
  --config configs/optimal.json \
  -v
```

### Documentación Rápida
```bash
python main.py -z proyecto.zip \
  -p "explica qué hace la clase UserManager" \
  --config configs/fast.json
```

### Búsqueda Técnica con Reranking ⭐ NUEVO
```bash
# Cross-Encoder: máxima precisión
python main.py -z backend.zip \
  -p "encuentra todas las funciones que manejan autenticación JWT" \
  --config configs/rerank_cross_encoder.json \
  -v

# MMR: diversidad sin redundancia
python main.py -z microservices.zip \
  -p "explica la arquitectura general del sistema" \
  --config configs/rerank_mmr.json \
  -v
```

## Comparación de Estrategias de Reranking

| Aspecto | Sin Reranking | Cross-Encoder | MMR |
|---------|---------------|---------------|-----|
| **Velocidad** | ⚡⚡⚡ Rápido | ⚡ Lento | ⚡⚡ Medio |
| **Precisión** | ⭐⭐ Buena | ⭐⭐⭐ Excelente | ⭐⭐ Buena |
| **Diversidad** | ⭐ Baja | ⭐ Baja | ⭐⭐⭐ Alta |
| **Costo computacional** | Bajo | Alto | Bajo |
| **Modelo adicional** | ❌ No | ✅ Sí | ❌ No |
| **Usa embeddings** | ✅ Sí | ❌ No | ✅ Sí |
| **Mejor para** | Queries simples | Queries técnicas precisas | Exploración amplia |

**Recomendación:**
- 🎯 **Cross-Encoder**: Para búsquedas críticas donde cada documento debe ser perfectamente relevante
- 🌈 **MMR**: Para obtener una visión completa evitando fragmentos similares
- ⚡ **Sin reranking**: Para desarrollo rápido o queries simples
