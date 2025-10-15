"""
Script principal para RAG sobre codebase.

Permite indexar código desde ZIP o GitHub y hacer queries usando RAG.
"""

import argparse
import sys
import os
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
    arg_parser.add_argument("-p", "--prompt", required=True, help="Pregunta sobre el código")
    arg_parser.add_argument("-c", "--collection-name", default="codeRAG", help="Nombre de la colección")
    arg_parser.add_argument("--config", help="Ruta al archivo de configuración JSON (ej: configs/optimal.json)")
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="Mostrar logs detallados del proceso RAG")

    args = arg_parser.parse_args()

    zip_path = args.zip
    github_link = args.github_link
    collection_name = args.collection_name
    user_prompt = args.prompt
    verbose = args.verbose
    config_path = args.config

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
    chroma.insert_docs(docs)
    print(f"✅ {len(docs)} fragmentos indexados correctamente")

    # ============================================================
    # FASE 3: CONSULTA RAG
    # ============================================================
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

    # Ejecutar RAG con logging mejorado
    res = chroma.rag(query=user_prompt, verbose=verbose)

    # ============================================================
    # RESULTADO FINAL
    # ============================================================
    print("\n" + "="*60)
    print("  RESPUESTA FINAL")
    print("="*60 + "\n")
    print(res)
    print("\n" + "="*60)