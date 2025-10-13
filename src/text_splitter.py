"""
Parseo y carga de archivos de código desde ZIP.

Funcionalidades:
- Parseo de archivos con LanguageParser (consciente del lenguaje)
- Extracción y procesamiento de archivos ZIP
- Filtrado automático de archivos soportados
"""

import os
import sys
import tempfile
from zipfile import ZipFile

from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document

from inserter import ChromaCollection
from utils.language_detector import is_supported, detect_language


def parse_file(filepath: str, parser_threshold: int = 3) -> list[Document]:
    """
    Parsea un archivo de código y lo divide en fragmentos.

    Args:
        filepath: Ruta al archivo a parsear
        parser_threshold: Tamaño mínimo de fragmentos (default: 3)

    Returns:
        Lista de Documents con fragmentos del código

    Note: Usa LanguageParser de Langchain con detección automática de lenguaje.
    """
    parser = LanguageParser(parser_threshold=parser_threshold)
    blob = Blob(path=filepath)
    docs = parser.lazy_parse(blob=blob)
    return list(docs)


def load_from_zipfile(zippath: str, parser_threshold: int = 3) -> list[Document]:
    """
    Carga y parsea todos los archivos soportados desde un ZIP.

    Args:
        zippath: Ruta al archivo ZIP
        parser_threshold: Tamaño mínimo de fragmentos de código

    Returns:
        Lista plana de Documents de todos los archivos parseados
    """
    all_documents = []

    # Crear directorio temporal para extraer archivos
    with tempfile.TemporaryDirectory() as temp_dir:
        with ZipFile(zippath, 'r') as zip_file:
            # Extraer todos los archivos
            zip_file.extractall(temp_dir)

            # Recorrer archivos extraídos
            for root, dirs, files in os.walk(temp_dir):
                # Filtrar directorios ocultos
                dirs[:] = [d for d in dirs if not d.startswith('.')]

                for filename in files:
                    # Saltar archivos ocultos
                    if filename.startswith('.'):
                        continue

                    filepath = os.path.join(root, filename)

                    # Filtrar solo archivos soportados
                    if not is_supported(filepath):
                        print(f"⊘ Saltando (no soportado): {filename}")
                        continue

                    try:
                        # Parsear archivo
                        print(f"✓ Parseando: {filename} ({detect_language(filepath)})")
                        docs = parse_file(filepath, parser_threshold=parser_threshold)

                        # Agregar metadata adicional
                        for doc in docs:
                            if doc.metadata is None:
                                doc.metadata = {}
                            doc.metadata['source_file'] = filename
                            doc.metadata['language'] = detect_language(filepath)

                        all_documents.extend(docs)
                        print(f"  → {len(docs)} fragmentos extraídos")

                    except Exception as e:
                        print(f"✗ Error parseando {filename}: {e}")
                        continue

    return all_documents


if __name__ == "__main__":
    """
    Script principal para indexar un codebase desde ZIP.

    Uso: python text_splitter.py <nombre_coleccion> <ruta_zip>
    Ejemplo: python text_splitter.py mi_proyecto ./codebase.zip
    """
    if len(sys.argv) < 3:
        print("Uso: python text_splitter.py <nombre_coleccion> <ruta_zip>")
        sys.exit(1)

    collection_name = sys.argv[1]
    zip_path = sys.argv[2]

    # Validar que existe el ZIP
    if not os.path.exists(zip_path):
        print(f"Error: No se encontró el archivo {zip_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Indexando codebase: {zip_path}")
    print(f"Colección: {collection_name}")
    print(f"{'='*60}\n")

    # Inicializar colección
    chroma = ChromaCollection(collection_name)

    # Cargar documentos desde ZIP
    print("Cargando archivos desde ZIP...")
    docs = load_from_zipfile(zip_path)

    if len(docs) == 0:
        print("\n✗ No se encontraron documentos para indexar")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Total de fragmentos a insertar: {len(docs)}")
    print(f"{'='*60}\n")

    # Insertar en ChromaDB
    print("Insertando en ChromaDB...")
    chroma.insert_docs(docs)
    print("✓ Documentos insertados exitosamente\n")

    # Hacer query de prueba
    test_query = "¿Qué funcionalidades tiene este código?"
    print(f"Query de prueba: '{test_query}'")
    print(f"{'='*60}\n")

    res = chroma.rag(query=test_query)
    print("Respuesta:")
    print(res)
    print(f"\n{'='*60}")