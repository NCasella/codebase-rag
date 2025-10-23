"""
Cargador de configuración JSON para el sistema RAG.

Permite cargar y validar archivos de configuración que definen
parámetros del modelo, retrieval, text splitting y embeddings.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptConfig:
    """Configuración de prompts."""
    template: str = "default"
    include_metadata: bool = True
    max_context_length: int = 8000


@dataclass
class ModelConfig:
    """Configuración del modelo LLM."""
    provider: str = "google"
    name: str = "gemini-2.5-flash-lite"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    top_p: float = 1.0


@dataclass
class RetrievalConfig:
    """Configuración de retrieval de documentos."""
    k_documents: int = 5
    include_metadata: bool = True
    similarity_threshold: Optional[float] = None


@dataclass
class TextSplittingConfig:
    """Configuración de división de texto."""
    parser_threshold: int = 3


@dataclass
class EmbeddingsConfig:
    """Configuración de embeddings."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    distance_function:str = "l2"
    device: str = "cpu"
    normalize_embeddings: bool = True


@dataclass
class RerankConfig:
    """Configuración de reranking."""
    enabled: bool = False
    strategy: str = "none"  # none, cross_encoder, mmr

    # Parámetros generales
    retrieve_k: int = 20  # Documentos a recuperar antes de reranking
    top_n: int = 5        # Documentos finales después de reranking

    # Cross-encoder specific
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    cross_encoder_device: str = "cpu"

    # MMR specific
    mmr_lambda: float = 0.5  # 0=máxima diversidad, 1=máxima relevancia


@dataclass
class RAGConfig:
    """
    Configuración completa del sistema RAG.

    Attributes:
        name: Nombre de la configuración
        description: Descripción breve
        prompt: Configuración de prompts
        model: Configuración del modelo LLM
        retrieval: Configuración de retrieval
        text_splitting: Configuración de text splitting
        embeddings: Configuración de embeddings
        rerank: Configuración de reranking

    Example:
        >>> config = RAGConfig.from_json("configs/optimal.json")
        >>> print(config.model.name)
        'gpt-4o'
    """
    name: str
    description: str
    prompt: PromptConfig
    model: ModelConfig
    retrieval: RetrievalConfig
    text_splitting: TextSplittingConfig
    embeddings: EmbeddingsConfig
    rerank: RerankConfig

    @classmethod
    def from_json(cls, json_path: str) -> 'RAGConfig':
        """
        Carga configuración desde archivo JSON.

        Args:
            json_path: Ruta al archivo JSON de configuración

        Returns:
            Instancia de RAGConfig con los valores cargados

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el JSON es inválido o faltan campos requeridos

        Example:
            >>> config = RAGConfig.from_json("configs/default.json")
        """
        path = Path(json_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Archivo de configuración no encontrado: {json_path}\n"
                f"Asegúrate de que el archivo existe en la ruta especificada."
            )

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido en {json_path}: {e}")

        # Validar campos requeridos (rerank es opcional)
        required_fields = ['name', 'description', 'prompt', 'model', 'retrieval',
                          'text_splitting', 'embeddings']
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValueError(f"Campos faltantes en config: {', '.join(missing)}")

        # Construir configuración con valores por defecto si faltan
        try:
            config = cls(
                name=data['name'],
                description=data['description'],
                prompt=PromptConfig(**data.get('prompt', {})),
                model=ModelConfig(**data.get('model', {})),
                retrieval=RetrievalConfig(**data.get('retrieval', {})),
                text_splitting=TextSplittingConfig(**data.get('text_splitting', {})),
                embeddings=EmbeddingsConfig(**data.get('embeddings', {})),
                rerank=RerankConfig(**data.get('rerank', {}))
            )
        except TypeError as e:
            raise ValueError(f"Error al parsear configuración: {e}")

        # Validar configuración
        config.validate()

        return config

    def validate(self) -> None:
        """
        Valida que los parámetros de configuración sean válidos.

        Raises:
            ValueError: Si algún parámetro es inválido
        """
        # Validar model
        valid_providers = ["google", "openai"]
        if self.model.provider.lower() not in valid_providers:
            raise ValueError(f"provider debe ser uno de {valid_providers}, got '{self.model.provider}'")

        if self.model.temperature < 0 or self.model.temperature > 2:
            raise ValueError(f"temperature debe estar entre 0 y 2, got {self.model.temperature}")

        if self.model.top_p < 0 or self.model.top_p > 1:
            raise ValueError(f"top_p debe estar entre 0 y 1, got {self.model.top_p}")

        if self.model.max_tokens is not None and self.model.max_tokens <= 0:
            raise ValueError(f"max_tokens debe ser positivo o None, got {self.model.max_tokens}")

        # Validar retrieval
        if self.retrieval.k_documents <= 0:
            raise ValueError(f"k_documents debe ser positivo, got {self.retrieval.k_documents}")

        if self.retrieval.similarity_threshold is not None:
            if self.retrieval.similarity_threshold < 0 or self.retrieval.similarity_threshold > 1:
                raise ValueError(f"similarity_threshold debe estar entre 0 y 1, got {self.retrieval.similarity_threshold}")

        # Validar text_splitting
        if self.text_splitting.parser_threshold <= 0:
            raise ValueError(f"parser_threshold debe ser positivo, got {self.text_splitting.parser_threshold}")

        # Validar prompt
        if self.prompt.max_context_length <= 0:
            raise ValueError(f"max_context_length debe ser positivo, got {self.prompt.max_context_length}")

        # Validar rerank
        valid_strategies = ["none", "cross_encoder", "mmr"]
        if self.rerank.strategy not in valid_strategies:
            raise ValueError(f"rerank.strategy debe ser uno de {valid_strategies}, got '{self.rerank.strategy}'")

        if self.rerank.retrieve_k <= 0:
            raise ValueError(f"rerank.retrieve_k debe ser positivo, got {self.rerank.retrieve_k}")

        if self.rerank.top_n <= 0:
            raise ValueError(f"rerank.top_n debe ser positivo, got {self.rerank.top_n}")

        if self.rerank.top_n > self.rerank.retrieve_k:
            raise ValueError(f"rerank.top_n ({self.rerank.top_n}) no puede ser mayor que retrieve_k ({self.rerank.retrieve_k})")

        if self.rerank.mmr_lambda < 0 or self.rerank.mmr_lambda > 1:
            raise ValueError(f"rerank.mmr_lambda debe estar entre 0 y 1, got {self.rerank.mmr_lambda}")

    def to_dict(self) -> dict:
        """
        Convierte la configuración a diccionario.

        Returns:
            Diccionario con todos los parámetros de configuración
        """
        return {
            'name': self.name,
            'description': self.description,
            'prompt': {
                'template': self.prompt.template,
                'include_metadata': self.prompt.include_metadata,
                'max_context_length': self.prompt.max_context_length
            },
            'model': {
                'provider': self.model.provider,
                'name': self.model.name,
                'temperature': self.model.temperature,
                'max_tokens': self.model.max_tokens,
                'top_p': self.model.top_p
            },
            'retrieval': {
                'k_documents': self.retrieval.k_documents,
                'include_metadata': self.retrieval.include_metadata,
                'similarity_threshold': self.retrieval.similarity_threshold
            },
            'text_splitting': {
                'parser_threshold': self.text_splitting.parser_threshold
            },
            'embeddings': {
                'model_name': self.embeddings.model_name,
                'device': self.embeddings.device,
                'normalize_embeddings': self.embeddings.normalize_embeddings
            },
            'rerank': {
                'enabled': self.rerank.enabled,
                'strategy': self.rerank.strategy,
                'retrieve_k': self.rerank.retrieve_k,
                'top_n': self.rerank.top_n,
                'cross_encoder_model': self.rerank.cross_encoder_model,
                'cross_encoder_device': self.rerank.cross_encoder_device,
                'mmr_lambda': self.rerank.mmr_lambda
            }
        }

    def __str__(self) -> str:
        """Representación legible de la configuración."""
        return (
            f"RAGConfig(name='{self.name}')\n"
            f"  • Prompt: {self.prompt.template}\n"
            f"  • Model: {self.model.provider}/{self.model.name} (temp={self.model.temperature})\n"
            f"  • Retrieval: k={self.retrieval.k_documents}\n"
            f"  • Embeddings: {self.embeddings.model_name}"
        )


def get_default_config() -> RAGConfig:
    """
    Retorna la configuración por defecto.

    Returns:
        RAGConfig con valores por defecto
    """
    return RAGConfig(
        name="default",
        description="Configuración por defecto",
        prompt=PromptConfig(),
        model=ModelConfig(),
        retrieval=RetrievalConfig(),
        text_splitting=TextSplittingConfig(),
        embeddings=EmbeddingsConfig(),
        rerank=RerankConfig()
    )
