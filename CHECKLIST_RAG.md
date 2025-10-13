# Checklist para Levantar el RAG

Esta checklist te guiará paso a paso para tener una versión básica funcional del Codebase RAG.

## 🔧 Fase 1: Configuración Inicial

- [x] **Verificar instalación de dependencias**
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Crear archivo `.env` con API Key de OpenAI**
  ```bash
  # En la raíz del proyecto crear .env con:
  OPENAI_API_KEY=tu_clave_aqui
  ```

- [ ] **Verificar que se puede importar ChromaDB**
  ```bash
  python -c "import chromadb; print('ChromaDB OK')"
  ```

- [ ] **Verificar que se puede importar OpenAI**
  ```bash
  python -c "from openai import OpenAI; print('OpenAI OK')"
  ```

---

## 🐛 Fase 2: Arreglar Bugs Críticos

### Bug 1: Query no se pasa como lista
- [x] **Archivo**: `src/inserter.py` línea 34
- [x] **Cambiar**: `documents,_=self.retrieve_k_similar_docs(query)`
- [x] **Por**: `documents,_=self.retrieve_k_similar_docs([query])`
- [x] **Razón**: El método espera una lista de queries

### Bug 2: Parse de archivos en ZIP
- [x] **Archivo**: `src/text_splitter.py` líneas 41-49
- [x] **Problema**: `parse_file(filename)` no puede acceder a archivos dentro del ZIP
- [x] **Solución**: Extraer temporalmente o leer el contenido del ZIP directamente
- [X] **Implementar una de estas opciones**:
  - [x] Opción A: Extraer ZIP a carpeta temporal
  - [ ] Opción B: Leer contenido en memoria y crear Blob desde bytes

### Bug 3: Integrar language_detector
- [x] **Archivo**: `src/text_splitter.py`
- [x] **Problema**: Usa su propio `_extension_map` en lugar del `language_detector.py`
- [x] **Solución**: Importar y usar `from src.utils.language_detector import is_supported, detect_language`

---

## 🧪 Fase 3: Crear Dataset de Prueba

- [x] **Crear carpeta `test_data/`** en la raíz del proyecto

- [x] **Crear 3-5 archivos Python simples** en `test_data/`:
  - [x] `main.py` - Script principal con función main
  - [x] `utils.py` - Funciones auxiliares (ej: suma, resta, formateo)
  - [x] `models.py` - Clases simples (ej: Usuario, Producto)
  - [x] `README.md` - Documentación básica del código
  - [x] `config.py` - Constantes y configuración

- [x] **Crear ZIP del dataset**
  ```bash
  cd test_data
  zip -r ../test_codebase.zip .
  cd ..
  ```

---

## 🔨 Fase 4: Crear Script de Prueba Básico

- [ ] **Crear `test_rag.py`** en la raíz del proyecto

- [ ] **Implementar test básico** que:
  - [ ] Carga archivos desde carpeta local (no ZIP)
  - [ ] Inserta documentos en ChromaDB
  - [ ] Hace una query simple
  - [ ] Imprime resultado

- [ ] **Ejemplo de estructura**:
  ```python
  # 1. Crear colección
  # 2. Cargar 1-2 archivos manualmente
  # 3. Insertar en ChromaDB
  # 4. Hacer query simple: "¿qué hace la función main?"
  # 5. Imprimir respuesta
  ```

---

## 🎯 Fase 5: Probar Pipeline Completo

### Test 1: Inserción de Documentos
- [ ] **Ejecutar script de prueba**
  ```bash
  python test_rag.py
  ```
- [ ] **Verificar que se crean documentos** (no errores de ChromaDB)
- [ ] **Verificar mensajes de debug** (ej: "X documentos insertados")

### Test 2: Retrieval
- [ ] **Hacer query de recuperación**
- [ ] **Verificar que devuelve documentos relevantes**
- [ ] **Imprimir los top 3 documentos más similares**

### Test 3: RAG Completo (Retrieval + Generation)
- [ ] **Hacer pregunta completa**
  - Ejemplos:
    - "¿Qué hace la función main?"
    - "¿Qué clases hay definidas?"
    - "Explica el propósito de utils.py"
- [ ] **Verificar que GPT responde coherentemente**
- [ ] **Verificar que la respuesta usa el contexto correcto**

---

## 🚀 Fase 6: Probar con ZIP (Objetivo Original)

- [ ] **Arreglar `load_from_zipfile()` con los bugs corregidos**

- [ ] **Probar con el ZIP de prueba**
  ```bash
  python src/text_splitter.py test_collection test_codebase.zip
  ```

- [ ] **Verificar que**:
  - [ ] Se extraen todos los archivos
  - [ ] Se parsean correctamente
  - [ ] Se insertan en ChromaDB
  - [ ] La query funciona

---

## ✅ Fase 7: Mejoras Básicas (Opcional pero Recomendado)

### Manejo de Errores
- [ ] **Agregar try-catch en parseo de archivos**
- [ ] **Validar existencia de `.env` al iniciar**
- [ ] **Manejar archivos que no se pueden parsear** (skip y continuar)
- [ ] **Agregar logs informativos** (ej: "Procesando archivo X...")

### Mensajes de Debug
- [ ] **Agregar contador de archivos procesados**
- [ ] **Mostrar archivos que se saltaron** (y por qué)
- [ ] **Mostrar tiempo de procesamiento**

### Validaciones
- [ ] **Verificar que el ZIP no está vacío**
- [ ] **Verificar que hay al menos 1 archivo soportado**
- [ ] **Validar que la colección se creó correctamente**

---

## 🎓 Fase 8: Preparación para el TP

- [ ] **Documentar en README**:
  - [ ] Cómo ejecutar el sistema
  - [ ] Ejemplos de queries
  - [ ] Screenshots o ejemplos de output

- [ ] **Crear notebook de demostración** (opcional):
  - [ ] Mostrar el proceso paso a paso
  - [ ] Incluir visualizaciones
  - [ ] Ejemplos de preguntas y respuestas

- [ ] **Preparar ejemplos de diferentes tipos de repos**:
  - [ ] Repo mono-lenguaje (solo Python)
  - [ ] Repo multi-lenguaje (Python + JS + Markdown)
  - [ ] Repo pequeño (< 10 archivos)
  - [ ] Repo mediano (20-50 archivos)

---

## 📊 Checklist de Verificación Final

Antes de considerar que tienes una versión funcionando, verifica:

- [ ] ✅ El sistema carga archivos desde ZIP sin errores
- [ ] ✅ Los documentos se insertan correctamente en ChromaDB
- [ ] ✅ Las queries retornan documentos relevantes
- [ ] ✅ GPT genera respuestas coherentes usando el contexto
- [ ] ✅ El sistema maneja errores básicos sin crashear
- [ ] ✅ Puedes hacer al menos 3 tipos de preguntas diferentes exitosamente
- [ ] ✅ El código está documentado con comentarios básicos
- [ ] ✅ Tienes al menos un dataset de prueba funcionando

---

## 🎯 Orden Recomendado de Ejecución

1. ✅ Configuración Inicial (Fase 1)
2. ✅ Arreglar Bug 1 y Bug 2 (Fase 2 - más fáciles)
3. ✅ Crear Dataset de Prueba (Fase 3)
4. ✅ Crear Script de Prueba sin ZIP (Fase 4)
5. ✅ Probar Retrieval + RAG (Fase 5)
6. ✅ Arreglar Bug 3 y Bug 4 (Fase 2 - más complejos)
7. ✅ Probar con ZIP completo (Fase 6)
8. ✅ Agregar mejoras básicas (Fase 7)
9. ✅ Preparar para presentación (Fase 8)

---

## 📝 Notas Importantes

- **No necesitas perfección**: Una versión básica funcionando es mejor que una versión perfecta sin terminar
- **Itera rápido**: Arregla un bug, prueba, continúa
- **Guarda versiones**: Haz commits frecuentes cuando algo funcione
- **Prioriza**: Los bugs 1 y 2 son 5 minutos, los bugs 3 y 4 son más complejos

---

## 🆘 Troubleshooting Común

### Error: "No module named 'chromadb'"
```bash
pip install chromadb
```

### Error: "API key not found"
- Verificar que existe `.env` en la raíz
- Verificar que la variable se llama exactamente `OPENAI_API_KEY`
- Reiniciar el script después de crear `.env`

### Error: "Model not found"
- Usar `gpt-4o-mini` o `gpt-3.5-turbo`
- Verificar que tu API key tiene acceso al modelo

### Error al parsear archivos
- Empezar solo con archivos Python (.py)
- Agregar más lenguajes gradualmente
- Usar try-catch para skip archivos problemáticos

---

¡Buena suerte! 🚀
