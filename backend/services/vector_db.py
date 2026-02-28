"""
services/vector_db.py — Wrapper sobre Qdrant + Abstracción de Embeddings.

Responsabilidades:
  1. Conectar con Qdrant (vector DB)
  2. Crear/verificar la colección de vectores
  3. Insertar chunks vectorizados
  4. Buscar por similitud coseno
  5. Generar embeddings (sentence-transformers u OpenAI)

Uso:
    from services.vector_db import VectorDBService

    vdb = VectorDBService()
    vdb.ensure_collection()
    vdb.upsert(chunks=["Hola mundo"], metadata=[{"source": "test.pdf"}])
    results = vdb.search("Hola", top_k=3)
"""

import uuid
import asyncio
from abc import ABC, abstractmethod

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
)

from core.config import settings


# ═══════════════════════════════════════════════════════════════
#  EMBEDDING PROVIDERS — Interfaz abstracta + implementaciones
# ═══════════════════════════════════════════════════════════════

class EmbeddingProvider(ABC):
    """Interfaz abstracta para generar embeddings."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts: Lista de strings a vectorizar.

        Returns:
            Lista de vectores (cada uno es una lista de floats).
        """
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Retorna la dimensión del vector de embedding."""
        ...


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Embeddings usando sentence-transformers (local, gratis).

    Modelo por defecto: all-MiniLM-L6-v2 (384 dims, ~80MB)
    Es rápido y suficientemente bueno para un hackathon.
    """

    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class OpenAIProvider(EmbeddingProvider):
    """
    Embeddings usando la API de OpenAI.

    Requiere OPENAI_API_KEY en el .env.
    Modelo por defecto: text-embedding-3-small (1536 dims)
    """

    def __init__(self, model_name: str = None):
        import openai

        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self._dim = settings.EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=self._model_name,
        )
        return [item.embedding for item in response.data]

    def dimension(self) -> int:
        return self._dim


from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """
    Factory — instancia el provider según la configuración.

    settings.EMBEDDING_PROVIDER puede ser:
      - "sentence-transformers" (por defecto, local, gratis)
      - "openai" (requiere API key)
    """
    if settings.EMBEDDING_PROVIDER == "openai":
        return OpenAIProvider()
    return SentenceTransformerProvider()


# ═══════════════════════════════════════════════════════════════
#  VECTOR DB SERVICE — Wrapper sobre Qdrant
# ═══════════════════════════════════════════════════════════════

class VectorDBService:
    """
    Servicio de base de datos vectorial.

    Encapsula toda la interacción con Qdrant y la generación de embeddings.
    Cada instancia crea su propio cliente Qdrant y su embedding provider.
    """

    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self.collection_name = settings.COLLECTION_NAME
        self.embedder = get_embedding_provider()

    def ensure_collection(self):
        """
        Crea la colección en Qdrant si no existe.
        Si ya existe, no hace nada (idempotente).
        """
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.dimension(),
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, chunks: list[str], metadata: list[dict]) -> int:
        """
        Vectoriza los chunks y los inserta en Qdrant.

        Args:
            chunks: Lista de textos a insertar.
            metadata: Lista de dicts con metadatos (source, page, etc.).
                      Debe tener la misma longitud que chunks.

        Returns:
            Número de puntos insertados.
        """
        if not chunks:
            return 0

        # Generar embeddings
        embeddings = self.embedder.embed(chunks)

        # Crear puntos para Qdrant
        points = []
        for i, (chunk, embedding, meta) in enumerate(zip(chunks, embeddings, metadata)):
            point = PointStruct(
                id=str(uuid.uuid4()),  # ID único por chunk
                vector=embedding,
                payload={
                    "text": chunk,      # El texto original del chunk
                    **meta,             # source, page, etc.
                },
            )
            points.append(point)

        # Insertar en batch
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return len(points)

    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        filters: dict = None,
    ) -> list[dict]:
        """
        Busca los chunks más similares a la query, con filtros dinámicos.

        Args:
            query: Texto de búsqueda.
            top_k: Número de resultados a devolver.
            filters: Diccionario de metadatos exactos a filtrar (ej. {"category": "RRHH"}).

        Returns:
            Lista de dicts con: text, score, source, y otros metadatos.
        """
        # Asegurarse de que la colección existe
        # (Idealmente en contexto async esto debería delegarse a un thread también, pero es muy rápido)
        await asyncio.to_thread(self.ensure_collection)

        # Vectorizar la query (CPU-Bound, bloquea el event loop. Lo mandamos a un thread)
        embeddings = await asyncio.to_thread(self.embedder.embed, [query])
        query_embedding = embeddings[0]

        # Construir filtros opcionales dinámicamente
        query_filter = None
        
        if filters:
            must_conditions = []
            
            for key, value in filters.items():
                if not value:
                    continue
                    
                if isinstance(value, list):
                    should_conditions = [
                        FieldCondition(key=key, match=MatchValue(value=v))
                        for v in value
                    ]
                    must_conditions.append(Filter(should=should_conditions))
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                    
            if must_conditions:
                query_filter = Filter(must=must_conditions)

        # Buscar en Qdrant (I/O-Bound síncrono, lo mandamos a un thread)
        results = await asyncio.to_thread(
            self.client.query_points,
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        # Formatear resultados
        formatted = []
        for point in results.points:
            formatted.append({
                "text": point.payload.get("text", ""),
                "score": point.score,
                "source": point.payload.get("source", "unknown"),
                "category": point.payload.get("category", "General"),
                "extension": point.payload.get("extension", ""),
                "page": point.payload.get("page", None),
                "chunk_index": point.payload.get("chunk_index", None),
            })

        return formatted
