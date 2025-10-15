# Plan de Implementación: Config System + Interactive Chat

Este documento detalla los pasos para implementar dos nuevas features:
1. **Sistema de configuración con archivos JSON**
2. **Chat interactivo con contexto conversacional**

---

## FASE 1: Sistema de Configuración JSON

### Objetivo
Permitir configurar todos los parámetros del sistema (modelo, retrieval, text splitting, embeddings) mediante archivos JSON, facilitando experimentación y reproducibilidad.

### Pasos de implementación

#### 1.1 Crear estructura de carpetas
```bash
mkdir configs
```

**Archivos a crear:**
- `configs/default.json` - Configuración balanceada
- `configs/optimal.json` - Mejor calidad (usa prompt optimal)
- `configs/fast.json` - Respuestas rápidas
- `configs/detailed.json` - Análisis profundo

#### 1.2 Diseñar schema JSON
**Archivo**: `configs/default.json`

```json
{
  "name": "default",
  "description": "Configuración balanceada para uso general",
  "prompt_template": "default",
  "model": {
    "name": "gpt-4.1-nano",
    "temperature": 0.1
  },
  "retrieval": {
    "k_documents": 5,
    "similarity_threshold": null
  },
  "text_splitting": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "embeddings": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

#### 1.3 Crear config loader
**Archivo nuevo**: `src/config_loader.py`

**Funcionalidades:**
- Clase `RAGConfig` con dataclass o pydantic
- Método `load_from_json(path)`
- Método `validate()` para validar parámetros
- Valores por defecto si falta algún campo
- Manejo de errores (archivo no existe, JSON inválido)

**Dependencias:**
```python
from dataclasses import dataclass
from pathlib import Path
import json
```

#### 1.4 Modificar ChromaCollection
**Archivo**: `src/inserter.py`

**Cambios necesarios:**
- Modificar `__init__()` para aceptar parámetro opcional `config: RAGConfig`
- Si se pasa `config`, usar sus valores en lugar de hardcoded
- Mantener retrocompatibilidad (parámetros individuales siguen funcionando)

**Ejemplo:**
```python
def __init__(
    self,
    collection_name: str,
    prompt_template: str = "default",
    config: RAGConfig | None = None
):
    if config:
        prompt_template = config.prompt_template
        self.model_config = config.model
        self.retrieval_config = config.retrieval
    # ...
```

#### 1.5 Modificar text_splitter
**Archivo**: `src/text_splitter.py`

**Cambios:**
- Modificar funciones para aceptar parámetros de chunk_size y chunk_overlap
- Actualmente están hardcodeados
- Pasar estos valores desde main.py usando la config

#### 1.6 Integrar en CLI
**Archivo**: `main.py`

**Cambios:**
- Agregar argumento `--config` al parser
- Cargar config si se especifica
- Pasar config a ChromaCollection
- Mostrar config activa en los logs de fase

**Ejemplo CLI:**
```bash
python main.py -z code.zip -p "pregunta" --config configs/optimal.json
```

#### 1.7 Testing
- Crear configs de prueba
- Verificar que cada config funciona
- Verificar retrocompatibilidad (sin --config sigue funcionando)

#### 1.8 Documentación
- Actualizar README.md con sección de configuración
- Documentar schema JSON
- Agregar ejemplos de configs

---

## FASE 2: Chat Interactivo con Contexto

### Objetivo
Crear interfaz REPL que permita conversación continua con el asistente, manteniendo contexto entre preguntas sin re-indexar.

### Pasos de implementación

#### 2.1 Crear ChatSession manager
**Archivo nuevo**: `src/chat_session.py`

**Funcionalidades:**
```python
class ChatSession:
    def __init__(self, chroma: ChromaCollection, config: RAGConfig):
        self.chroma = chroma
        self.config = config
        self.history = []  # Lista de {"role": "user/assistant", "content": "..."}
        self.max_history = 10  # Sliding window

    def add_user_message(self, content: str) -> None:
        """Agregar mensaje del usuario al historial"""

    def add_assistant_message(self, content: str) -> None:
        """Agregar respuesta del asistente al historial"""

    def get_history_context(self) -> list[dict]:
        """Retornar últimos N mensajes para contexto"""

    def clear_history(self) -> None:
        """Limpiar historial de conversación"""

    def ask(self, query: str, verbose: bool = False) -> str:
        """
        Hacer pregunta con contexto de historial.
        1. Agregar query al historial
        2. Hacer retrieval de documentos relevantes
        3. Construir mensajes con historial + documentos
        4. Llamar a OpenAI
        5. Agregar respuesta al historial
        6. Retornar respuesta
        """
```

#### 2.2 Modificar ChromaCollection para contexto
**Archivo**: `src/inserter.py`

**Opción A (simple)**:
- Crear método `rag_with_history(query, history, verbose)`
- Similar a `rag()` pero acepta lista de mensajes previos
- Retrieval usa solo query actual
- Generation usa historial completo

**Opción B (avanzado)**:
- Implementar query reformulation usando historial
- Antes de retrieval, reformular query considerando contexto
- Más preciso pero más complejo y costoso

**Recomendación**: Empezar con Opción A

#### 2.3 Crear interfaz REPL
**Archivo**: `main.py` (agregar modo chat)

**Funcionalidades:**
- Detectar flag `--chat` en CLI
- Si está activo, entrar en loop interactivo después de indexar
- Input loop con `input()` o librería `prompt_toolkit` (más avanzado)
- Comandos especiales:
  - `exit` o `quit` - Salir
  - `/clear` - Limpiar historial
  - `/history` - Mostrar historial
  - `/config` - Mostrar configuración activa
  - `/help` - Ayuda de comandos

**Pseudo-código:**
```python
if args.chat:
    session = ChatSession(chroma, config)
    print("Modo chat activado. Escribe 'exit' para salir.\n")

    while True:
        try:
            user_input = input("> ").strip()

            if user_input in ["exit", "quit"]:
                break

            if user_input.startswith("/"):
                handle_command(user_input, session)
                continue

            if not user_input:
                continue

            response = session.ask(user_input, verbose=args.verbose)
            print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\n\nSaliendo...")
            break
        except Exception as e:
            print(f"Error: {e}")
```

#### 2.4 Mejorar UX del chat
**Mejoras opcionales:**

**2.4.1 Colorización de output**
- Usar `rich` o `colorama`
- User input en un color, respuesta en otro
- Comandos especiales resaltados

**2.4.2 Better input handling**
- Usar `prompt_toolkit` para:
  - Historial de comandos (flecha arriba)
  - Auto-completado
  - Multi-línea
  - Sintaxis highlighting

**2.4.3 Indicadores visuales**
- Spinner mientras piensa
- Progress bar para retrieval
- Typing indicator

#### 2.5 Implementar sliding window
**Archivo**: `src/chat_session.py`

**Problema**: Historial muy largo = demasiados tokens = costoso y lento

**Solución**:
- Mantener solo últimos N mensajes (ej: 10)
- Opcionalmente: system message con resumen de conversación antigua
- Configurar N en config JSON

#### 2.6 Manejo de contexto en retrieval
**Consideración importante:**

Cuando el usuario pregunta:
```
User: "¿Qué hace la función login?"
Assistant: "La función login valida credenciales..."
User: "¿Y qué pasa si falla?"  ← "falla" se refiere a login
```

**Opciones:**

**A) Retrieval solo de query actual (más simple)**
- Busca documentos relevantes a "¿Y qué pasa si falla?"
- Puede no encontrar documentos relevantes
- Pero el historial en generation ayuda

**B) Query reformulation (más preciso)**
```python
def reformulate_query(query: str, history: list) -> str:
    # Usar LLM para reformular query considerando historial
    # Input: "¿Y qué pasa si falla?" + historial
    # Output: "¿Qué pasa si la función login falla?"
    return reformulated_query
```
- Más costoso (llamada extra a LLM)
- Mejor retrieval
- Implementar en Fase 2B (después)

#### 2.7 Testing del chat
- Probar conversaciones de varios turnos
- Verificar que el contexto se mantiene
- Probar comandos especiales
- Verificar que sliding window funciona
- Testear con diferentes configs

#### 2.8 Documentación
- Actualizar README con modo chat
- Documentar comandos especiales
- Agregar ejemplos de conversaciones
- Explicar límites de contexto

---

## Orden de Implementación Recomendado

### Sprint 1: Config System (estimado: 2-3 horas)
```
1. Crear configs/ con JSONs
2. Implementar src/config_loader.py
3. Modificar src/inserter.py para usar configs
4. Modificar src/text_splitter.py para usar configs
5. Integrar en main.py
6. Testing básico
7. Actualizar README
```

### Sprint 2: Chat Básico (estimado: 3-4 horas)
```
1. Implementar src/chat_session.py (sin sliding window aún)
2. Agregar método rag_with_history() a ChromaCollection
3. Crear loop REPL básico en main.py
4. Implementar comandos /exit, /clear, /history
5. Testing de conversaciones multi-turn
```

### Sprint 3: Chat Avanzado (estimado: 2-3 horas)
```
1. Implementar sliding window en ChatSession
2. Agregar colorización con rich/colorama
3. Mejorar UX con indicadores visuales
4. Agregar más comandos útiles (/config, /help, /docs)
5. Testing exhaustivo
6. Documentación completa
```

### Sprint 4 (Opcional): Query Reformulation
```
1. Implementar reformulate_query() en ChatSession
2. Comparar resultados con/sin reformulation
3. Hacer configurable en JSON
4. Documentar trade-offs
```

---

## Dependencias Nuevas

Agregar a `requirements.txt`:
```
rich>=13.0.0          # Para colorización y mejor UX
prompt_toolkit>=3.0   # Para mejor input handling (opcional)
```

---

## Decisiones de Diseño

### ¿Por qué JSON para configs?
- ✅ Legible y editable manualmente
- ✅ Estándar ampliamente usado
- ✅ Fácil de versionar en git
- ❌ Alternativas: YAML (más complejo), TOML (menos común en ML)

### ¿Por qué no usar LangChain ConversationChain?
- ✅ Control total sobre el historial
- ✅ Más simple y transparente
- ✅ Menos dependencias
- ❌ Pero se podría migrar después si se necesita

### ¿Sliding window vs. Summarization?
- **Sliding window**: Mantener últimos N mensajes
  - ✅ Simple, rápido, predecible
  - ❌ Pierde contexto antiguo completamente
- **Summarization**: Resumir historial antiguo
  - ✅ Mantiene contexto antiguo
  - ❌ Costoso, complejo, puede perder detalles
- **Decisión**: Empezar con sliding window, evaluar summarization después

---

## Checklist de Implementación

### Fase 1: Config System
- [ ] Crear carpeta `configs/`
- [ ] Crear `configs/default.json`
- [ ] Crear `configs/optimal.json`
- [ ] Crear `configs/fast.json`
- [ ] Crear `configs/detailed.json`
- [ ] Implementar `src/config_loader.py`
- [ ] Modificar `ChromaCollection.__init__()` para usar config
- [ ] Modificar `text_splitter.py` para usar config
- [ ] Agregar flag `--config` al CLI
- [ ] Testing de configs
- [ ] Actualizar README

### Fase 2: Chat Interactivo
- [ ] Implementar `src/chat_session.py`
- [ ] Agregar `rag_with_history()` a `ChromaCollection`
- [ ] Crear loop REPL en `main.py`
- [ ] Implementar comando `/exit`
- [ ] Implementar comando `/clear`
- [ ] Implementar comando `/history`
- [ ] Implementar comando `/config`
- [ ] Implementar comando `/help`
- [ ] Implementar sliding window
- [ ] Agregar colorización con `rich`
- [ ] Testing de conversaciones
- [ ] Actualizar README con modo chat

### Fase 3 (Opcional): Mejoras avanzadas
- [ ] Implementar query reformulation
- [ ] Agregar `prompt_toolkit` para mejor input
- [ ] Agregar spinners y progress bars
- [ ] Implementar logging de sesiones
- [ ] Agregar export de conversaciones

---

## Preguntas para decidir

Antes de empezar, definir:

1. **¿Qué parámetros incluir en config?**
   - ¿Solo los mencionados o más?
   - ¿Incluir parámetros de ChromaDB (distance function, etc)?

2. **¿Sliding window size?**
   - ¿10 mensajes? ¿20? ¿Configurable?

3. **¿Librerías de UX?**
   - ¿Usar `rich` para colorización?
   - ¿Usar `prompt_toolkit` para input mejorado?
   - ¿O mantenerlo simple con input() básico?

4. **¿Query reformulation?**
   - ¿Implementarlo desde el inicio o dejarlo para después?

5. **¿Verbose en chat?**
   - ¿Mostrar documentos recuperados en cada query?
   - ¿Hacer que sea toggleable con `/verbose`?

---

## Resultado Final Esperado

### CLI con config:
```bash
python main.py -z code.zip --config configs/optimal.json -p "¿Qué hace main.py?"
# Usa configuración optimal (mejor calidad)
```

### CLI con chat interactivo:
```bash
python main.py -z code.zip --config configs/fast.json --chat

============================================================
  CODEBASE RAG - MODO CHAT INTERACTIVO
============================================================

📦 Indexando: code.zip
📊 Colección: codeRAG
🎯 Config: configs/fast.json (gpt-3.5-turbo, k=3)

✅ Listo! 250 fragmentos indexados

Comandos: /help /clear /history /config /exit

> ¿Qué hace la función login?

La función login en auth.py:42 valida las credenciales del usuario
contra la base de datos y retorna un JWT si son válidas...

> ¿Y qué pasa si las credenciales son incorrectas?

Si las credenciales son incorrectas, la función retorna un status 401
con el mensaje "Invalid credentials"...

> /history

Historial (2 mensajes):
1. User: ¿Qué hace la función login?
2. Assistant: La función login en auth.py:42...
3. User: ¿Y qué pasa si las credenciales son incorrectas?
4. Assistant: Si las credenciales son incorrectas...

> exit

¡Hasta luego!
```

---

## Estimación Total

- **Fase 1 (Config System)**: 2-3 horas
- **Fase 2 (Chat Básico)**: 3-4 horas
- **Fase 3 (Chat Avanzado)**: 2-3 horas
- **Total**: 7-10 horas

**Recomendación**: Implementar en orden, validar cada fase antes de continuar.
