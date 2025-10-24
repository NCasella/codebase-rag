"""
Script principal para RAG sobre codebase.

Permite indexar código desde ZIP o GitHub y hacer queries usando RAG.
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path
from src.inserter import ChromaCollection
from src.text_splitter import load_from_zipfile, load_from_github_link
from src.config_loader import RAGConfig

if __name__ == "__main__":
    # Configurar argumentos CLI
    arg_parser = argparse.ArgumentParser(
        prog="codebase RAG",
        description="Sistema RAG para consultas sobre código"
    )
    group = arg_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-z", "--zip", help="Ruta al archivo ZIP con el código")
    group.add_argument("-g", "--github_link", help="URL del repositorio de GitHub")
    arg_parser.add_argument("-p", "--prompt", help="Pregunta sobre el código (modo single-shot). Si se omite, entra en modo chat interactivo")
    arg_parser.add_argument("-c", "--collection-name", default="codeRAG", help="Nombre de la colección")
    arg_parser.add_argument("--config", help="Ruta al archivo de configuración JSON (ej: configs/optimal.json)")
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="Mostrar logs detallados del proceso RAG")
    arg_parser.add_argument("--conversation-id", help="ID de conversación anterior para continuar el diálogo (de un archivo output/*.txt previo)")

    args = arg_parser.parse_args()

    zip_path = args.zip
    github_link = args.github_link
    collection_name = args.collection_name
    user_prompt = args.prompt
    verbose = args.verbose
    config_path = args.config
    conversation_id = args.conversation_id

    # Cargar configuración si se especifica
    config = None
    if config_path:
        try:
            config = RAGConfig.from_json(config_path)
            print(f"\n⚙️  Configuración cargada: {config.name}")
            print(f"   {config.description}")
        except Exception as e:
            print(f"\n✗ Error al cargar configuración: {e}")
            sys.exit(1)

    # ============================================================
    # FASE 1: CARGA DE DATOS
    # ============================================================
    print("\n" + "="*60)
    print("  CODEBASE RAG - FASE 1: CARGA DE DATOS")
    print("="*60)

    # Obtener parser_threshold de config si existe
    parser_threshold = config.text_splitting.parser_threshold if config else 3

    if zip_path:
        print(f"\n📦 Fuente: Archivo ZIP")
        print(f"   Ruta: {zip_path}")
        if not os.path.exists(zip_path):
            print(f"\n✗ Error: No se encontró el archivo {zip_path}")
            sys.exit(1)
        print(f"\n⏳ Extrayendo y parseando archivos...")
        print(f"   (parser_threshold={parser_threshold})")
        docs = load_from_zipfile(zip_path, parser_threshold=parser_threshold)
    else:
        print(f"\n🔗 Fuente: GitHub")
        print(f"   URL: {github_link}")
        print(f"\n⏳ Descargando repositorio...")
        print(f"   (parser_threshold={parser_threshold})")
        try:
            docs = load_from_github_link(github_link, parser_threshold=parser_threshold)
        except Exception as e:
            print(f"\n✗ Error: {e}")
            sys.exit(1)

    if len(docs) == 0:
        print("\n✗ No se encontraron documentos para indexar")
        sys.exit(1)

    print(f"\n✅ Carga completada:")
    print(f"   • {len(docs)} fragmentos de código extraídos")

    # ============================================================
    # FASE 2: INDEXACIÓN (EMBEDDINGS + CHROMADB)
    # ============================================================
    print("\n" + "="*60)
    print("  FASE 2: INDEXACIÓN EN BASE DE DATOS VECTORIAL")
    print("="*60)

    print(f"\n📊 Colección: '{collection_name}'")
    if config:
        print(f"⏳ Inicializando ChromaDB con config '{config.name}'...")
        print(f"   • Prompt: {config.prompt.template}")
        print(f"   • Modelo: {config.model.name}")
        print(f"   • Embeddings: {config.embeddings.model_name}")
        chroma = ChromaCollection(collection_name, config=config)
    else:
        print(f"⏳ Inicializando ChromaDB con configuración por defecto...")
        chroma = ChromaCollection(collection_name)
    print(f"✅ ChromaDB inicializado")

    print(f"\n⏳ Generando embeddings y almacenando en ChromaDB...")
    print(f"   (Esto puede tardar unos segundos para {len(docs)} fragmentos)")
    stats = chroma.insert_docs(docs)

    # Mostrar estadísticas de inserción
    if stats['duplicates'] > 0:
        print(f"✅ Indexación completada:")
        print(f"   • Total de fragmentos: {stats['total']}")
        print(f"   • Duplicados omitidos: {stats['duplicates']}")
        print(f"   • Nuevos insertados: {stats['inserted']}")
    else:
        print(f"✅ {stats['inserted']} fragmentos indexados correctamente")

    # ============================================================
    # FASE 3: CONSULTA RAG o MODO CHAT
    # ============================================================

    if user_prompt:
        # MODO SINGLE-SHOT: Una query y salir
        print("\n" + "="*60)
        print("  FASE 3: CONSULTA RAG (RETRIEVAL + GENERATION)")
        print("="*60)

        print(f"\n❓ Pregunta del usuario:")
        print(f"   '{user_prompt}'")

        if config and verbose:
            print(f"\n📋 Configuración activa:")
            print(f"   • Recuperar: {config.retrieval.k_documents} documentos")
            print(f"   • Modelo: {config.model.name} (temp={config.model.temperature})")
            print(f"   • Max context: {config.prompt.max_context_length} caracteres")

        if verbose:
            print(f"\n⏳ Paso 1/3: Buscando fragmentos relevantes...")
            print(f"   (Búsqueda semántica en {len(docs)} fragmentos)")

        if conversation_id:
            if verbose:
                print(f"\n🔄 Continuando conversación: {conversation_id}")

        # Ejecutar RAG con logging mejorado
        res, response_obj = chroma.rag(query=user_prompt, verbose=verbose, conversation_id=conversation_id)
        with open(f"output/{collection_name}.txt","a") as f:
            f.write(f"Q: {user_prompt}\n")
            f.write(f"A: {res}\n")
            f.write("="*30)

        # ============================================================
        # RESULTADO FINAL
        # ============================================================
        print("\n" + "="*60)
        print("  RESPUESTA FINAL")
        print("="*60 + "\n")
        print(res)
        print("\n" + "="*60)

        # ============================================================
        # GUARDAR CONVERSATION ID PARA CONTINUACIÓN
        # ============================================================
        if response_obj and response_obj.conversation_id:
            # Crear directorio output si no existe
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{timestamp}.txt"

            # Guardar conversation_id
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response_obj.conversation_id)

            print(f"\n💾 Conversation ID guardado en: {output_file}")
            print(f"   Para continuar esta conversación, use: --conversation-id {response_obj.conversation_id}")
        elif conversation_id:
            print(f"\n⚠️  Nota: El proveedor actual no soporta continuación de conversación")

    else:
        # MODO CHAT INTERACTIVO: Loop infinito hasta /exit
        from src.chat import start_chat
        start_chat(chroma, config if config else chroma.config)