"""Modelos y enums para estrategias de reranking."""

from enum import Enum


class RerankStrategy(Enum):
    """Estrategias de reranking disponibles."""
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    MMR = "mmr"
