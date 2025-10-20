"""
Gestión de ChromaDB y RAG para consultas sobre código.

Funcionalidades:
- Almacenar fragmentos de código con embeddings vectoriales
- Búsqueda semántica de documentos similares
- Generación de respuestas contextuales con el LLM a elección
"""
import os
import chromadb
from src.llm.factory import create_llm, Provider, GoogleModel, OpenAIModel
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from uuid import uuid4
from typing import Optional
from .prompt_loader import get_prompt_loader
from .config_loader import RAGConfig, EmbeddingsConfig, get_default_config
from .plotter import plot_relevant_docs
import umap
import numpy as np 
from tqdm import tqdm
# Cargar variables de entorno (.env)
_=load_dotenv(find_dotenv())

# Función de embedding global: SentenceTransformer basado en BERT
# Convierte texto a vectores de alta dimensionalidad para búsqueda semántica
embedding_function = SentenceTransformerEmbeddingFunction()

# Cliente global de ChromaDB (base de datos vectorial)
_chroma_client = chromadb.Client()


class ChromaCollection():
    """
    Gestiona una colección de ChromaDB con capacidades RAG.

    Attributes:
        _chroma_collection: Colección de ChromaDB
        llm_client: Cliente de LLM (Google Gemini u OpenAI)

    Example:
        >>> collection = ChromaCollection('mi_proyecto')
        >>> collection.insert_docs([Document(page_content="def hello(): pass")])
        >>> respuesta = collection.rag("¿cómo se define una función?")
    """

    def __init__(
        self,
        collection_name: str,
        prompt_template: str = "default",
        config: Optional[RAGConfig] = None
    ) -> None:
        """
        Inicializa la colección de ChromaDB y el cliente LLM.

        Args:
            collection_name: Nombre único para la colección
            prompt_template: Nombre del template de prompt a usar (ignorado si se pasa config)
            config: Configuración RAG completa (opcional)

        Raises:
            ValueError: Si el prompt_template no existe o config es inválida

        Example:
            >>> # Opción 1: Sin config (usa valores por defecto)
            >>> collection = ChromaCollection("proyecto")

            >>> # Opción 2: Con prompt_template específico
            >>> collection = ChromaCollection("proyecto", prompt_template="spanish")

            >>> # Opción 3: Con config completa
            >>> config = RAGConfig.from_json("configs/optimal.json")
            >>> collection = ChromaCollection("proyecto", config=config)
        """
        # Si se pasa config, usarla; sino usar valores por defecto
        if config is None:
            self.config = get_default_config()
            self.config.prompt.template = prompt_template
        else:
            self.config = config

        # Inicializar ChromaDB con embeddings configurados
        self.chroma_collection = _initialize_collection(
            collection_name=collection_name,
            embeddings_config=self.config.embeddings
        )
        # Configurar provider y modelo desde config
        provider_str = self.config.model.provider.lower()
        model_name = self.config.model.name

        # Determinar provider enum
        try:
            provider_enum = Provider(provider_str)
        except ValueError:
            # Fallback a Google con advertencia
            print(f"⚠️  Advertencia: Provider '{provider_str}' no válido. Usando fallback: Google Gemini")
            provider_enum = Provider.GOOGLE
            model_name = GoogleModel.GEMINI_2_5_FLASH_LITE

        # Determinar API key según provider
        if provider_enum == Provider.GOOGLE:
            api_key_env = "GEMINI_API_KEY"
        elif provider_enum == Provider.OPENAI:
            api_key_env = "OPENAI_API_KEY"
        else:
            api_key_env = None

        # Obtener API key o usar fallback
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if not api_key:
            print(f"⚠️  Advertencia: {api_key_env} no encontrado. Usando fallback: Google Gemini")
            provider_enum = Provider.GOOGLE
            model_name = GoogleModel.GEMINI_2_5_FLASH_LITE
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "No se encontró ninguna API key válida. "
                    "Configure GEMINI_API_KEY o OPENAI_API_KEY en su .env"
                )

        # Crear cliente LLM
        self.llm_client = create_llm(
            provider=provider_enum,
            model=model_name,
            api_key=api_key
        )
        self.prompt_loader = get_prompt_loader()

        # Validar que el prompt existe
        available = self.prompt_loader.list_prompts()
        if self.config.prompt.template not in available:
            raise ValueError(
                f"Prompt template '{self.config.prompt.template}' no encontrado.\n"
                f"Disponibles: {', '.join(available)}"
            )

        self.prompt_template = self.config.prompt.template

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
        results = self.chroma_collection.query(query_texts=[query], n_results=k, include=['metadatas','documents', 'embeddings'])
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

        self.chroma_collection.add(
            ids=[str(uuid4()) for _ in docs],
            documents=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs]
        )
    
    def rag(self, query: str, model: Optional[str] = None, verbose: bool = False) -> str:
        """
        Responde preguntas sobre el código usando RAG (Retrieval-Augmented Generation).

        Pipeline: RETRIEVAL → AUGMENTATION → GENERATION

        Args:
            query: Pregunta sobre el código (ej: "¿Cómo funciona la autenticación?")
            model: Modelo de LLM (opcional, usa el de config si no se especifica)
            verbose: Si es True, muestra logs detallados del proceso

        Returns:
            Respuesta generada con contexto del código

        Note: Usa parámetros de self.config para k_documents, temperature, etc.
        """
        model_name = model if model is not None else self.config.model.name

        k = self.config.retrieval.k_documents
        documents, results = self.retrieve_k_similar_docs(query, k=k)

        if self.config.retrieval.similarity_threshold is not None:
            # TODO: Implementar filtrado por similarity threshold
            # Requiere acceso a distances de ChromaDB
            pass

        if verbose:
            print(f"✅ Encontrados {len(documents)} fragmentos relevantes")
            print(f"\n📄 Fragmentos recuperados:")
            for i, doc in enumerate(documents, 1):
                preview = doc[:100].replace('\n', ' ') + "..." if len(doc) > 100 else doc.replace('\n', ' ')
                print(f"   {i}. {preview}")

        # AUGMENTATION: Unir documentos en un solo contexto
        information = "\n".join(documents)

        # Limitar contexto según max_context_length de config
        max_length = self.config.prompt.max_context_length
        if len(information) > max_length:
            information = information[:max_length]
            if verbose:
                print(f"⚠️  Contexto truncado a {max_length} caracteres")

        if verbose:
            print(f"\n⏳ Paso 2/3: Construyendo prompt con contexto...")
            print(f"   • Longitud del contexto: {len(information)} caracteres")
            print(f"   • Fragmentos incluidos: {len(documents)}")

        # Cargar prompt desde archivo
        system_prompt = self.prompt_loader.load(self.prompt_template)

        # Construir prompt con system + user message
        # TODO sanitize! UNICODE 
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nInformation:\n{information}"
            }
        ]
        self.project_and_plot_relevant_docs(query=query, title=query, k_similar_results=results)
        # GENERATION: Llamar al LLM para generar respuesta
        if verbose:
            print(f"\n⏳ Paso 3/3: Generando respuesta con {self.llm_client.model}...")
            print(f"   • Temperature: {self.config.model.temperature}")
            print(f"   • Tokens aproximados en contexto: ~{len(information) // 4}")

        # Construir parámetros del LLM desde config
        llm_params = {
            # "model": model_name,
            "messages": messages,
            "temperature": self.config.model.temperature,
            "top_p": self.config.model.top_p
        }

        # Agregar max_tokens solo si está configurado
        if self.config.model.max_tokens is not None:
            llm_params["max_tokens"] = self.config.model.max_tokens

        response = self.llm_client.generate(
            messages=messages,
            temperature=self.config.model.temperature,
            top_p=self.config.model.top_p,
        )
        # response = self.openai_client.chat.completions.create(**openai_params)

        if verbose:
            print(f"✅ Respuesta generada exitosamente")
            if response.usage:
                prompt_tok = response.usage.get('prompt_tokens', 'N/A')
                completion_tok = response.usage.get('completion_tokens', 'N/A')
                total_tok = response.usage.get('total_tokens', 'N/A')
                print(f"   • Tokens usados: {total_tok} total ({prompt_tok} prompt + {completion_tok} completion)")

        content = response.text
        return content

    def project_and_plot_relevant_docs(self,query, title, k_similar_results):
        embeddings=self.chroma_collection.get(include=['embeddings'])['embeddings']
        umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)

        
        query_embedding = embedding_function([query])[0]
        retrieved_embeddings = k_similar_results['embeddings'][0]
        projected_query_embedding = project_embeddings([query_embedding], umap_transform)
        projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)
        projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)
        plot_relevant_docs(projected_dataset_embeddings, projected_query_embedding, projected_retrieved_embeddings, query, title)


def _initialize_collection(
    collection_name: str,
    embeddings_config: Optional[EmbeddingsConfig] = None
) -> chromadb.Collection:
    """
    Crea o recupera una colección de ChromaDB (idempotente).

    Args:
        collection_name: Nombre único de la colección
        embeddings_config: Configuración de embeddings (opcional)

    Returns:
        Instancia de la colección

    Note: Si ya existe, la recupera. Si no existe, la crea con embedding function configurado.
    """
    existing_collections = _chroma_client.list_collections()
    existing_collection_names = [collection.name for collection in existing_collections]

    if collection_name not in existing_collection_names:
        # Crear embedding function según config
        if embeddings_config is not None:
            emb_function = SentenceTransformerEmbeddingFunction(
                model_name=embeddings_config.model_name,
                device=embeddings_config.device
            )
            distance_function=embeddings_config.distance_function
        else:
            # Usar función de embedding por defecto
            emb_function = embedding_function
            distance_function="l2"

        # Crear nueva colección con función de embedding
        chroma_collection = _chroma_client.create_collection(
            name=collection_name,
            embedding_function=emb_function,configuration={
                "hnsw":{
                    "space":distance_function
                    }
            }
        )
    else:
        chroma_collection = _chroma_client.get_collection(name=collection_name)

    return chroma_collection
def project_embeddings(embeddings, umap_transform):
  """
  Projects high-dimensional embeddings into a 2D space using UMAP.

  Parameters:
  embeddings (numpy.ndarray): A 2D array-like object containing the embeddings to be transformed,
                              where each row represents an embedding vector.
  umap_transform (umap.UMAP): A pre-trained UMAP model to perform the transformation.

  Returns:
  numpy.ndarray: A 2D array where each row is a 2D embedding resulting from the UMAP transformation.
  """
  umap_embeddings = np.empty((len(embeddings),2))   # Mappeamos desde la longitud de nuestros embeddings a 2D

  for i, embedding in enumerate(tqdm(embeddings)):
      umap_embeddings[i] = umap_transform.transform([embedding])

  return umap_embeddings