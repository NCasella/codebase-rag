"""
Gestión de ChromaDB y RAG para consultas sobre código.

Funcionalidades:
- Almacenar fragmentos de código con embeddings vectoriales
- Búsqueda semántica de documentos similares
- Generación de respuestas contextuales con OpenAI
"""

import os
import chromadb
from openai import OpenAI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from uuid import uuid4

# Cargar variables de entorno (.env)
_=load_dotenv(find_dotenv())

# Función de embedding global: SentenceTransformer basado en BERT
# Convierte texto a vectores de alta dimensionalidad para búsqueda semántica
_embedding_function = SentenceTransformerEmbeddingFunction()

# Cliente global de ChromaDB (base de datos vectorial)
_chroma_client = chromadb.Client()


class ChromaCollection():
    """
    Gestiona una colección de ChromaDB con capacidades RAG.

    Attributes:
        _chroma_collection: Colección de ChromaDB
        openai_client: Cliente de OpenAI

    Example:
        >>> collection = ChromaCollection('mi_proyecto')
        >>> collection.insert_docs([Document(page_content="def hello(): pass")])
        >>> respuesta = collection.rag("¿cómo se define una función?")
    """

    def __init__(self, collection_name: str) -> None:
        """
        Inicializa la colección de ChromaDB y el cliente de OpenAI.

        Args:
            collection_name: Nombre único para la colección

        TODO: Permitir diferentes clientes LLM (local, Anthropic, etc.)
        """
        self._chroma_collection = _initialize_collection(collection_name=collection_name)
        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def retrieve_k_similar_docs(self, query: str, k: int = 5) -> tuple[list[str], dict]:
        """
        Recupera los k documentos más similares a la query.

        Args:
            query: Consulta en lenguaje natural
            k: Número de documentos a recuperar (default: 5)

        Returns:
            (documentos, resultados): Lista de textos y resultados completos de ChromaDB

        TODO: Optimización batch - Procesar múltiples queries a la vez
              Cambiar a queries: list[str] y retornar [[docs_q1], [docs_q2], ...]
        """
        results = self._chroma_collection.query(query_texts=[query], n_results=k, include=['documents', 'embeddings'])
        retrieved_documents = results['documents'][0]
        return retrieved_documents, results
    
    def insert_docs(self, docs: list[Document]) -> None:
        """
        Inserta documentos en ChromaDB con embeddings automáticos.

        Args:
            docs: Lista de Documents con page_content y metadata

        Example:
            >>> docs = [Document(page_content="def suma(a, b): return a + b",
            ...                  metadata={"file": "utils.py"})]
            >>> collection.insert_docs(docs)

        Note: Genera IDs con UUID4. Si la lista está vacía, no hace nada.
        """
        docs_list = list(docs or [])
        if len(docs_list) == 0:
            print("docs vacio")
            return

        self._chroma_collection.add(
            ids=[str(uuid4()) for _ in docs],
            documents=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs]
        )
    
    def rag(self, query: str, model: str = "gpt-4.1-nano") -> str:
        """
        Responde preguntas sobre el código usando RAG (Retrieval-Augmented Generation).

        Pipeline: RETRIEVAL → AUGMENTATION → GENERATION

        Args:
            query: Pregunta sobre el código (ej: "¿Cómo funciona la autenticación?")
            model: Modelo de OpenAI (default: gpt-4.1-nano)

        Returns:
            Respuesta generada con contexto del código

        Note: Recupera 5 documentos similares y los usa como contexto.
        """
        # RETRIEVAL: Buscar los 5 documentos más similares
        documents, _ = self.retrieve_k_similar_docs(query)

        # AUGMENTATION: Unir documentos en un solo contexto
        information = "\n".join(documents)

        # Construir prompt con system + user message
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert agent in coding. Users will ask questions involving a code base. "
                    "You will be shown the user's question and the relevant code involving the question. "
                    "Use the information given to answer the user's question."
                )
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nInformation:\n{information}"
            }
        ]

        # GENERATION: Llamar a OpenAI para generar respuesta
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages
        )

        content = response.choices[0].message.content
        return content

def _initialize_collection(collection_name: str) -> chromadb.Collection:
    """
    Crea o recupera una colección de ChromaDB (idempotente).

    Args:
        collection_name: Nombre único de la colección

    Returns:
        Instancia de la colección

    Note: Si ya existe, la recupera. Si no existe, la crea con _embedding_function.
    """
    existing_collections = _chroma_client.list_collections()
    existing_collection_names = [collection.name for collection in existing_collections]

    if collection_name not in existing_collection_names:
        # Crear nueva colección con función de embedding
        chroma_collection = _chroma_client.create_collection(
            name=collection_name,
            embedding_function=_embedding_function
        )
    else:
        # Recuperar colección existente
        chroma_collection = _chroma_client.get_collection(name=collection_name)

    return chroma_collection
