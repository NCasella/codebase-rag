"""
Script de consultas para RAG sobre codebase.

Permite realizar queries sobre colecciones de ChromaDB ya indexadas.
Este script solo realiza consultas (Fase 3).
Para indexación, usar index.py.
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path
from src.inserter import ChromaCollection
from src.config_loader import RAGConfig

if __name__ == "__main__":
    # Configurar argumentos CLI
    arg_parser = argparse.ArgumentParser(
        prog="codebase RAG - Query",
        description="Consulta una colección de ChromaDB ya indexada"
    )
    arg_parser.add_argument("-c", "--collection-name", default="codeRAG", help="Nombre de la colección")
    arg_parser.add_argument("-p", "--prompt", help="Pregunta sobre el código (modo single-shot). Si se omite, entra en modo chat interactivo")
    arg_parser.add_argument("--config", help="Ruta al archivo de configuración JSON (ej: configs/optimal.json)")
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="Mostrar logs detallados del proceso RAG")
    arg_parser.add_argument("--conversation-id", help="ID de conversación anterior para continuar el diálogo (de un archivo output/*.txt previo)")

    args = arg_parser.parse_args()

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
    # INICIALIZACIÓN DE COLECCIÓN
    # ============================================================
    print("\n" + "="*60)
    print("  CODEBASE RAG - INICIALIZACIÓN")
    print("="*60)

    print(f"\n📊 Colección: '{collection_name}'")
    if config:
        print(f"⏳ Cargando colección existente con config '{config.name}'...")
        print(f"   • Prompt: {config.prompt.template}")
        print(f"   • Modelo: {config.model.name}")
        print(f"   • Embeddings: {config.embeddings.model_name}")
        chroma = ChromaCollection(collection_name, config=config)
    else:
        print(f"⏳ Cargando colección existente con configuración por defecto...")
        chroma = ChromaCollection(collection_name)
    print(f"✅ Colección cargada correctamente")

    # Verificar que la colección tiene documentos
    try:
        collection_size = chroma.chroma_collection.count()
        if collection_size == 0:
            print(f"\n⚠️  Advertencia: La colección '{collection_name}' está vacía")
            print(f"   Debe indexar documentos primero usando:")
            print(f"   python index.py -z <archivo.zip> -c {collection_name}")
            sys.exit(1)
        print(f"   • Documentos en colección: {collection_size}")
    except Exception as e:
        print(f"\n✗ Error: No se pudo acceder a la colección '{collection_name}'")
        print(f"   {e}")
        print(f"\n   ¿La colección existe? Use index.py para crearla primero.")
        sys.exit(1)

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
            print(f"   (Búsqueda semántica en {collection_size} fragmentos)")

        if conversation_id:
            if verbose:
                print(f"\n🔄 Continuando conversación: {conversation_id}")

        # Ejecutar RAG con logging mejorado
        res, response_obj = chroma.rag(query=user_prompt, verbose=verbose, conversation_id=conversation_id)
        os.mkdir("output") if not os.path.exists("output") else None
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
