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
    Range,
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
        range_filters: dict = None,
        exact_filters: dict = None,
    ) -> list[dict]:
        """
        Busca los chunks más similares a la query, con filtros dinámicos.

        Args:
            query: Texto de búsqueda.
            top_k: Número de resultados a devolver.
            filters: Diccionario de metadatos exactos a filtrar (ej. {"category": "RRHH"}).
            range_filters: Diccionario de metadatos de rango a filtrar (ej. {"file_size_bytes": {"gte": 1000, "lte": 5000}}).

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
        must_conditions = []
        
        if filters:
            # En vez de "must" (AND) estricto entre extensiones y categorías,
            # lo cambiamos a "should" (OR) global. Si el usuario marca "PDF" y "Finanzas",
            # quiere ver todos los PDFs y además todos los docs de Finanzas.
            should_conditions = []
            
            for key, value in filters.items():
                if not value:
                    continue
                    
                if isinstance(value, list):
                    for v in value:
                        should_conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
                else:
                    should_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                    
            if should_conditions:
                must_conditions.append(Filter(should=should_conditions))
                
        if exact_filters:
            for key, value in exact_filters.items():
                if value is not None:
                    # MatchValue is case-sensitive exact match. For Author it might be better to use MatchText or handle it case-insensitively if needed, but since Qdrant handles case-sensitive MatchValue by default we will stick to MatchText if it's a full-text index, or MatchValue. Since 'author' is not explicitly full-text indexed, an exact MatchValue or lowercase search is best.
                    # We will use MatchValue for exact match, as we want complete names "John Doe"
                    must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                
        if range_filters:
            for key, r_val in range_filters.items():
                gte_val = r_val.get("gte")
                lte_val = r_val.get("lte")
                if gte_val is not None or lte_val is not None:
                    must_conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(
                                gte=float(gte_val) if gte_val is not None else None,
                                lte=float(lte_val) if lte_val is not None else None
                            )
                        )
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
                "exif_metadata": point.payload.get("exif_metadata", None),
            })

        return formatted

    async def get_by_source(self, source: str) -> list[dict]:
        """
        Recupera TODOS los chunks indexados de un archivo fuente.
        Usa scroll (no search) para no necesitar un vector query.
        """
        await asyncio.to_thread(self.ensure_collection)

        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        )

        results, _ = await asyncio.to_thread(
            self.client.scroll,
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=100,
            with_payload=True,
        )

        formatted = []
        for point in results:
            formatted.append({
                "text": point.payload.get("text", ""),
                "source": point.payload.get("source", "unknown"),
                "category": point.payload.get("category", "General"),
                "extension": point.payload.get("extension", ""),
                "page": point.payload.get("page", None),
                "chunk_index": point.payload.get("chunk_index", None),
                "author": point.payload.get("author", None),
                "creator": point.payload.get("creator", None),
                "subject": point.payload.get("subject", None),
                "keywords": point.payload.get("keywords", None),
                "producer": point.payload.get("producer", None),
                "exif_metadata": point.payload.get("exif_metadata", None),
            })

        # Ordenar por chunk_index para reconstruir el documento en orden
        formatted.sort(key=lambda x: x.get("chunk_index", 0) or 0)
        return formatted
