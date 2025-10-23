"""
Parseo y carga de archivos de código desde ZIP.

Funcionalidades:
- Parseo de archivos con LanguageParser (consciente del lenguaje)
- Extracción y procesamiento de archivos ZIP
- Filtrado automático de archivos soportados
"""
import io
import requests
import os
import sys
import tempfile
from zipfile import ZipFile

from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document

from src.utils.language_detector import is_supported, detect_language


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
                            doc.metadata['language'] = detect_language(filepath)
                            doc.metadata["source"]= extract_from_zipname(filepath , zippath)

                        all_documents.extend(docs)
                        print(f"  → {len(docs)} fragmentos extraídos")

                    except Exception as e:
                        print(f"✗ Error parseando {filename}: {e}")
                        continue

    return all_documents

def load_from_github_link(github_link: str, parser_threshold: int = 3) -> list[Document]:
    """
    Descarga y parsea un repositorio de GitHub.

    Args:
        github_link: URL del repositorio de GitHub
        parser_threshold: Tamaño mínimo de fragmentos de código

    Returns:
        Lista plana de Documents de todos los archivos parseados
    """
    repo_url=github_link.rstrip("/")
    if not repo_url.endswith(".zip"):
        repo_url=repo_url.replace("github.com", "api.github.com/repos") + "/zipball/"
    response = requests.get(repo_url)
    response.raise_for_status()

    zip_bytes=io.BytesIO(response.content)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        tmp_zip.write(response.content)
        tmp_zip_path=tmp_zip.name

    return load_from_zipfile(tmp_zip_path, parser_threshold=parser_threshold)
    
def extract_from_zipname(path: str, zip_filename: str) -> str:
    zip_name = os.path.splitext(os.path.basename(zip_filename))[0]
    parts = path.split('/')
    if zip_name in parts:
        idx = parts.index(zip_name)
        return '/'.join(parts[idx+1:])
    return path