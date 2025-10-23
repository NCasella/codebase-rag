"""Clase base abstracta para rerankers."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class BaseReranker(ABC):
    """Clase base para todos los rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[dict]] = None,
        top_n: Optional[int] = None
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Reordena documentos según relevancia.

        Args:
            query: La query del usuario
            documents: Lista de documentos recuperados
            embeddings: Embeddings de los documentos (opcional, usado por MMR)
            metadata: Metadatos de los documentos (opcional)
            top_n: Número de documentos a retornar después de reranking

        Returns:
            Tupla de:
                - Lista de documentos reordenados
                - Lista de índices originales
                - Lista de scores de relevancia
        """
        pass
