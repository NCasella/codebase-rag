"""
Script de indexación para RAG sobre codebase.

Permite indexar código desde ZIP o GitHub en ChromaDB.
Este script solo realiza la indexación (Fases 1-2).
Para queries, usar query.py.
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
        prog="codebase RAG - Indexer",
        description="Indexa código en ChromaDB para consultas posteriores"
    )
    group = arg_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-z", "--zip", help="Ruta al archivo ZIP con el código")
    group.add_argument("-g", "--github_link", help="URL del repositorio de GitHub")
    arg_parser.add_argument("-c", "--collection-name", default="codeRAG", help="Nombre de la colección")
    arg_parser.add_argument("--config", help="Ruta al archivo de configuración JSON (ej: configs/optimal.json)")

    args = arg_parser.parse_args()

    zip_path = args.zip
    github_link = args.github_link
    collection_name = args.collection_name
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
    # INFORMACIÓN DE USO
    # ============================================================
    print("\n" + "="*60)
    print("  INDEXACIÓN COMPLETADA")
    print("="*60)
    print(f"\nPara realizar consultas sobre esta colección, ejecute:")
    print(f"  python query.py -c {collection_name} -p \"su pregunta aquí\"")
    print(f"\nO entre en modo chat interactivo:")
    print(f"  python query.py -c {collection_name}")
    if config_path:
        print(f"\nPara usar la misma configuración, agregue: --config {config_path}")
    print("\n" + "="*60)
