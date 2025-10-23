"""Reranker basado en Cross-Encoder para scoring preciso query-documento."""

from typing import List, Tuple, Optional
import logging
from ..base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Reranker que usa un modelo Cross-Encoder para calcular scores de relevancia.

    Cross-encoders procesan (query, documento) juntos y dan un score de relevancia
    más preciso que la similitud de embeddings, pero son más lentos.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", device: str = "cpu"):
        """
        Inicializa el CrossEncoderReranker.

        Args:
            model_name: Nombre del modelo de HuggingFace a usar
            device: Dispositivo para inferencia ('cpu' o 'cuda')
        """
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, device=device)
            self.model_name = model_name
            logger.info(f"CrossEncoderReranker inicializado con modelo {model_name} en {device}")
        except ImportError:
            raise ImportError(
                "sentence-transformers no está instalado. "
                "Instálalo con: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Error al cargar modelo {model_name}: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[dict]] = None,
        top_n: Optional[int] = None
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Reordena documentos usando Cross-Encoder.

        Args:
            query: La query del usuario
            documents: Lista de documentos recuperados
            embeddings: No usado por CrossEncoder
            metadata: Metadatos (no usado)
            top_n: Número de documentos a retornar después de reranking

        Returns:
            Tupla de:
                - Lista de documentos reordenados por relevancia
                - Lista de índices originales
                - Lista de scores de relevancia
        """
        if not documents:
            return [], [], []

        # Crear pares (query, documento) para el cross-encoder
        pairs = [[query, doc] for doc in documents]

        # Calcular scores de relevancia
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            scores = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            logger.error(f"Error al calcular scores con CrossEncoder: {e}")
            # Fallback: retornar en orden original
            n = top_n if top_n is not None else len(documents)
            return documents[:n], list(range(n)), [1.0] * n

        # Ordenar por score descendente (mayor score = más relevante)
        scored_docs = list(zip(documents, scores, range(len(documents))))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Seleccionar top_n
        n = top_n if top_n is not None else len(documents)
        n = min(n, len(scored_docs))

        reranked_docs = [doc for doc, _, _ in scored_docs[:n]]
        original_indices = [idx for _, _, idx in scored_docs[:n]]
        reranked_scores = [score for _, score, _ in scored_docs[:n]]

        logger.debug(
            f"Reranked {len(documents)} docs to top {n}. "
            f"Score range: [{min(reranked_scores):.3f}, {max(reranked_scores):.3f}]"
        )

        return reranked_docs, original_indices, reranked_scores
