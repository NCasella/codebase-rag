"""Reranker basado en MMR (Maximal Marginal Relevance) para diversidad."""

from typing import List, Tuple, Optional
import logging
import numpy as np
from ..base import BaseReranker

logger = logging.getLogger(__name__)


class MMRReranker(BaseReranker):
    """
    Reranker que usa MMR para balancear relevancia y diversidad.

    MMR (Maximal Marginal Relevance) selecciona documentos que son:
    1. Relevantes a la query (alta similitud)
    2. Diversos entre sí (baja similitud con docs ya seleccionados)

    Formula: MMR = lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected_docs))
    """

    def __init__(self, lambda_param: float = 0.5):
        """
        Inicializa el MMRReranker.

        Args:
            lambda_param: Balance entre relevancia (1.0) y diversidad (0.0)
                         0.0 = máxima diversidad, 1.0 = máxima relevancia
        """
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError(f"lambda debe estar entre 0.0 y 1.0, recibido: {lambda_param}")

        self.lambda_param = lambda_param
        logger.info(f"MMRReranker inicializado con lambda={lambda_param}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _compute_mmr_scores(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: List[np.ndarray],
        selected_indices: List[int]
    ) -> List[float]:
        """
        Calcula scores MMR para todos los documentos no seleccionados.

        Args:
            query_embedding: Embedding de la query
            doc_embeddings: Embeddings de todos los documentos
            selected_indices: Índices de documentos ya seleccionados

        Returns:
            Lista de scores MMR para cada documento
        """
        mmr_scores = []

        for i, doc_emb in enumerate(doc_embeddings):
            if i in selected_indices:
                mmr_scores.append(-float('inf'))  # Ya seleccionado
                continue

            # Similitud con la query (relevancia)
            query_sim = self._cosine_similarity(query_embedding, doc_emb)

            # Similitud máxima con documentos ya seleccionados (redundancia)
            if selected_indices:
                max_sim_to_selected = max(
                    self._cosine_similarity(doc_emb, doc_embeddings[j])
                    for j in selected_indices
                )
            else:
                max_sim_to_selected = 0.0

            # Score MMR
            mmr_score = (
                self.lambda_param * query_sim
                - (1 - self.lambda_param) * max_sim_to_selected
            )
            mmr_scores.append(mmr_score)

        return mmr_scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[dict]] = None,
        top_n: Optional[int] = None
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Reordena documentos usando MMR para diversidad.

        Args:
            query: La query del usuario (no usada directamente, pero se asume
                   que el primer embedding es el de la query si está incluido)
            documents: Lista de documentos recuperados
            embeddings: Embeddings de los documentos (REQUERIDO para MMR)
            metadata: Metadatos (no usado)
            top_n: Número de documentos a retornar después de reranking

        Returns:
            Tupla de:
                - Lista de documentos reordenados por MMR
                - Lista de índices originales
                - Lista de scores MMR
        """
        if not documents:
            return [], [], []

        if embeddings is None or len(embeddings) == 0:
            logger.warning(
                "MMR requiere embeddings. Retornando documentos en orden original."
            )
            n = top_n if top_n is not None else len(documents)
            return documents[:n], list(range(n)), [1.0] * n

        # Verificar que tenemos embeddings para todos los documentos
        # Nota: ChromaDB puede incluir el embedding de la query como primer elemento
        # o puede que solo tenga embeddings de documentos. Asumimos que embeddings
        # corresponden 1:1 con documents
        if len(embeddings) < len(documents):
            logger.warning(
                f"Menos embeddings ({len(embeddings)}) que documentos ({len(documents)}). "
                f"Usando documentos disponibles."
            )
            documents = documents[:len(embeddings)]

        # Convertir embeddings a numpy arrays
        try:
            doc_embeddings = [np.array(emb) for emb in embeddings[:len(documents)]]
        except Exception as e:
            logger.error(f"Error al convertir embeddings: {e}")
            n = top_n if top_n is not None else len(documents)
            return documents[:n], list(range(n)), [1.0] * n

        # Para MMR, necesitamos el embedding de la query
        # Asumimos que ChromaDB no lo incluye, así que calculamos similitudes
        # basándonos solo en las similitudes entre documentos y su orden original
        # (que ya está ordenado por similitud a la query)

        # El primer documento es el más similar a la query, lo usamos como proxy
        # o asumimos que el orden original ya refleja similitud a query
        query_sim_proxy = list(range(len(documents), 0, -1))  # Decreciente

        n = top_n if top_n is not None else len(documents)
        n = min(n, len(documents))

        # Algoritmo MMR greedy
        selected_indices = []
        selected_scores = []

        for _ in range(n):
            if len(selected_indices) == 0:
                # Primer documento: seleccionar el más relevante (índice 0)
                selected_indices.append(0)
                selected_scores.append(query_sim_proxy[0])
            else:
                # Calcular MMR scores para documentos no seleccionados
                mmr_scores = []
                for i in range(len(documents)):
                    if i in selected_indices:
                        mmr_scores.append(-float('inf'))
                        continue

                    # Similitud a query (proxy: orden original inverso normalizado)
                    query_sim = query_sim_proxy[i] / len(documents)

                    # Similitud máxima a documentos seleccionados
                    max_sim_to_selected = max(
                        self._cosine_similarity(doc_embeddings[i], doc_embeddings[j])
                        for j in selected_indices
                    )

                    # Score MMR
                    mmr_score = (
                        self.lambda_param * query_sim
                        - (1 - self.lambda_param) * max_sim_to_selected
                    )
                    mmr_scores.append(mmr_score)

                # Seleccionar documento con mayor MMR score
                best_idx = int(np.argmax(mmr_scores))
                selected_indices.append(best_idx)
                selected_scores.append(mmr_scores[best_idx])

        # Construir resultado
        reranked_docs = [documents[i] for i in selected_indices]
        reranked_scores = selected_scores

        logger.debug(
            f"MMR reranked {len(documents)} docs to top {n}. "
            f"Lambda={self.lambda_param}, "
            f"Score range: [{min(reranked_scores):.3f}, {max(reranked_scores):.3f}]"
        )

        return reranked_docs, selected_indices, reranked_scores
