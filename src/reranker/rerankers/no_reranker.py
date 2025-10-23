"""Reranker que no hace reranking - pass-through."""

from typing import List, Tuple, Optional
from ..base import BaseReranker


class NoReranker(BaseReranker):
    """Reranker que mantiene el orden original (sin reranking)."""

    def __init__(self):
        """Inicializa el NoReranker."""
        pass

    def rerank(
        self,
        query: str,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[dict]] = None,
        top_n: Optional[int] = None
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Retorna los documentos en su orden original.

        Args:
            query: La query del usuario (no usado)
            documents: Lista de documentos
            embeddings: Embeddings (no usado)
            metadata: Metadatos (no usado)
            top_n: Número de documentos a retornar

        Returns:
            Tupla de (documentos, índices, scores dummy de 1.0)
        """
        n = top_n if top_n is not None else len(documents)
        n = min(n, len(documents))

        selected_docs = documents[:n]
        indices = list(range(n))
        scores = [1.0] * n  # Scores dummy

        return selected_docs, indices, scores
