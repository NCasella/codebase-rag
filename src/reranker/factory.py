"""Factory para crear instancias de rerankers según configuración."""

import logging
from typing import TYPE_CHECKING

from .models import RerankStrategy
from .base import BaseReranker
from .rerankers import NoReranker, CrossEncoderReranker, MMRReranker

if TYPE_CHECKING:
    from ..config_loader import RerankConfig

logger = logging.getLogger(__name__)


class RerankFactory:
    """Factory para crear rerankers basados en configuración."""

    @staticmethod
    def create_reranker(config: 'RerankConfig') -> BaseReranker:
        """
        Crea una instancia de reranker según la configuración.

        Args:
            config: Configuración de reranking

        Returns:
            Instancia de BaseReranker según la estrategia configurada

        Raises:
            ValueError: Si la estrategia no es válida
        """
        try:
            strategy = RerankStrategy(config.strategy)
        except ValueError:
            logger.error(f"Estrategia de reranking inválida: {config.strategy}")
            raise ValueError(
                f"Estrategia de reranking '{config.strategy}' no válida. "
                f"Opciones: {[s.value for s in RerankStrategy]}"
            )

        if strategy == RerankStrategy.NONE:
            logger.info("Usando NoReranker (sin reranking)")
            return NoReranker()

        elif strategy == RerankStrategy.CROSS_ENCODER:
            logger.info(
                f"Creando CrossEncoderReranker con modelo {config.cross_encoder_model}"
            )
            return CrossEncoderReranker(
                model_name=config.cross_encoder_model,
                device=config.cross_encoder_device
            )

        elif strategy == RerankStrategy.MMR:
            logger.info(f"Creando MMRReranker con lambda={config.mmr_lambda}")
            return MMRReranker(lambda_param=config.mmr_lambda)

        else:
            raise ValueError(f"Estrategia no implementada: {strategy}")
