"""
services/vector_db.py — EL ARCHIVERO ESPACIAL (Base de Datos Vectorial).
----------------------------------------------------------------------
Este módulo es el que permite que la app "entienda" lo que buscas. En lugar 
de buscar palabras exactas, buscamos por significado.

¿CÓMO FUNCIONA? (En 3 pasos):
1. EMBEDDINGS: Convertimos el texto en una lista de números (coordenadas). 
   Textos con significados similares (ej: "perro" y "can") acaban "cerca" 
   geométricamente.
2. QDRANT: Es nuestro almacén de coordenadas. Guarda los trozos de texto 
   junto con su ubicación en ese mapa de significados.
3. BÚSQUEDA HÍBRIDA: Cuando preguntas algo, buscamos "cerca" de tu pregunta 
   pero también miramos palabras exactas. ¡Es lo mejor de ambos mundos!
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
    TextIndexParams,
    TokenizerType,
    PayloadSchemaType,
)

from core.config import settings


# ═══════════════════════════════════════════════════════════════
#  EMBEDDING PROVIDERS — Interfaz abstracta + implementaciones
# ═══════════════════════════════════════════════════════════════

class EmbeddingProvider(ABC):
    """Interfaz abstracta para homogeneizar diversos motores de vectorización."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Transforma una lista de cadenas de texto en sus correspondientes vectores.

        Args:
            texts (list[str]): Fragmentos de texto a procesar.

        Returns:
            list[list[float]]: Lista de vectores de alta dimensión.
        """
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Retorna la dimensión del vector de embedding."""
        ...


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Proveedor de embeddings local basado en la librería Sentence-Transformers.

    Ideal para entornos desconectados o para minimizar costes, ejecutando
    los modelos directamente en la CPU/GPU del servidor.
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
    Servicio principal de interacción con la Base de Datos Vectorial.

    Encapsula la complejidad técnica de Qdrant y la lógica de negocio 
    relacionada con la persistencia de fragmentos de documentos.
    Utiliza el patrón Singleton indirecto mediante el embedding provider configurado.
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
        Garantiza la existencia y configuración óptima de la colección en Qdrant.

        Este método es idempotente. Si la colección no existe:
            1. La crea con la dimensión adecuada según el embedding provider.
            2. Configura la métrica de distancia COSINE (estándar para texto).
            3. Inicializa un índice de carga (payload index) sobre el campo 'text' 
               con tokenización por palabras para habilitar búsquedas léxicas rápidas.
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

        # Crear text index sobre el campo 'text' para búsqueda léxica
        # Si ya existe, Qdrant lo ignora silenciosamente (idempotente)
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=40,
                    lowercase=True,
                ),
            )
        except Exception:
            pass  # Índice ya existe o no se pudo crear

    def upsert(self, chunks: list[str], metadata: list[dict]) -> int:
        """
        Transforma fragmentos de texto en vectores y los persiste en Qdrant.

        Proceso:
            1. Vectorización: Envía los chunks al embedder (Local o API).
            2. Estructuración: Empaqueta cada vector con su ID único (UUIDv4) 
               y sus metadatos asociados.
            3. Inserción: Realiza una operación de 'upsert' masiva por eficiencia.

        Args:
            chunks (list[str]): Textos limpios para indexar.
            metadata (list[dict]): Metadatos enriquecidos (fuente, página, etc.).

        Returns:
            int: Cantidad de fragmentos insertados exitosamente.
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
        role: str = "normal",
    ) -> list[dict]:
        """
        Ejecuta una búsqueda puramente semántica (vectorial) con filtros dinámicos.

        Utiliza la similitud del coseno para encontrar los fragmentos que más se 
        aproximan al significado de la consulta. Implementa seguridad a nivel de 
        registro (RBAC) ocultando automáticamente documentos 'admin_only'.

        Args:
            query (str): Pregunta o términos del usuario.
            top_k (int): Cantidad máxima de resultados.
            filters (dict): Filtros de categoría o extensión (operadores OR internos).
            range_filters (dict): Filtros numéricos para fechas o tamaños.
            exact_filters (dict): Coincidencias exactas (ej. autores).
            role (str): Rol del usuario para control de visibilidad.

        Returns:
            list[dict]: Lista de resultados formateados con texto y score de relevancia.
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
                    must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                
        if range_filters:
            for key, bounds in range_filters.items():
                range_args = {}
                if "gte" in bounds and bounds["gte"] is not None:
                    range_args["gte"] = float(bounds["gte"])
                if "lte" in bounds and bounds["lte"] is not None:
                    range_args["lte"] = float(bounds["lte"])
                
                if range_args:
                    must_conditions.append(FieldCondition(key=key, range=Range(**range_args)))
                    
        if must_conditions:
             query_filter = Filter(must=must_conditions)

        # ── Payload filtering por visibilidad (RBAC) ──────────────
        # Los documentos marcados como visibility="admin_only" solo los
        # puede ver el rol admin. Cualquier otro rol los excluye vía must_not.
        if role != "admin":
            visibility_condition = FieldCondition(
                key="visibility", match=MatchValue(value="admin_only")
            )
            if query_filter:
                existing_must_not = list(query_filter.must_not or [])
                query_filter = Filter(
                    must=query_filter.must,
                    should=query_filter.should,
                    must_not=existing_must_not + [visibility_condition],
                )
            else:
                query_filter = Filter(must_not=[visibility_condition])
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

    async def text_search(
        self,
        query: str,
        top_k: int = 100,
        filters: dict = None,
        range_filters: dict = None,
        exact_filters: dict = None,
        role: str = "normal",
    ) -> list[dict]:
        """
        Realiza una búsqueda léxica literal (estilo Grep / Ctrl+F).

        A diferencia de la búsqueda semántica, este método busca coincidencias 
        exactas de caracteres utilizando el índice de texto de Qdrant. Es ideal 
        para encontrar códigos técnicos, IDs de empleados o términos muy específicos.

        Calcula un score sintético basado en la frecuencia de aparición del término.
        """
        await asyncio.to_thread(self.ensure_collection)

        # Construir filtros
        must_conditions = [
            FieldCondition(key="text", match=MatchText(text=query))
        ]

        if filters:
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
                    must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        if range_filters:
            for key, bounds in range_filters.items():
                range_args = {}
                if "gte" in bounds and bounds["gte"] is not None:
                    range_args["gte"] = float(bounds["gte"])
                if "lte" in bounds and bounds["lte"] is not None:
                    range_args["lte"] = float(bounds["lte"])
                if range_args:
                    must_conditions.append(FieldCondition(key=key, range=Range(**range_args)))

        # RBAC: excluir docs admin_only para roles no-admin
        must_not_conditions = []
        if role != "admin":
            must_not_conditions.append(
                FieldCondition(key="visibility", match=MatchValue(value="admin_only"))
            )

        query_filter = Filter(
            must=must_conditions,
            must_not=must_not_conditions if must_not_conditions else None,
        )

        # Scroll para obtener todos los chunks que contienen el texto
        all_points = []
        offset = None
        while True:
            result = await asyncio.to_thread(
                self.client.scroll,
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            points, next_offset = result
            all_points.extend(points)
            if next_offset is None or len(all_points) >= top_k:
                break
            offset = next_offset

        # Calcular un score sintético basado en frecuencia de la query en el texto
        query_lower = query.lower()
        formatted = []
        for point in all_points[:top_k]:
            text = point.payload.get("text", "")
            # Score = frecuencia normalizada del término en el chunk
            count = text.lower().count(query_lower)
            score = min(1.0, count * 0.15 + 0.50) if count > 0 else 0.30

            formatted.append({
                "text": text,
                "score": score,
                "source": point.payload.get("source", "unknown"),
                "category": point.payload.get("category", "General"),
                "extension": point.payload.get("extension", ""),
                "page": point.payload.get("page", None),
                "chunk_index": point.payload.get("chunk_index", None),
                "exif_metadata": point.payload.get("exif_metadata", None),
            })

        # Ordenar por score descendente
        formatted.sort(key=lambda r: r["score"], reverse=True)
        return formatted

    async def hybrid_search(
        self,
        query: str,
        query_text: str,
        top_k: int = 5,
        filters: dict = None,
        range_filters: dict = None,
        exact_filters: dict = None,
        role: str = "lector",
    ) -> list[dict]:
        """
        EL SÚPER BUSCADOR (Hybrid Search).
        ----------------------------------
        Este es el secreto de la precisión de Meiga. Combina dos técnicas:
        
        1. BÚSQUEDA SEMÁNTICA (GPS): Busca por concepto. Encuentra "coche" 
           si buscas "vehículo".
        2. BÚSQUEDA LÉXICA (Lupa): Busca la palabra exacta. Encuentra el número 
           de serie "XYZ-123".
        
        ¿POR QUÉ AMBAS?
        La semántica es genial para conversar, pero a veces falla con nombres propios 
        o códigos técnicos. Al mezclarlas, Meiga es inteligente y precisa a la vez.
        """
        await asyncio.to_thread(self.ensure_collection)

        # Vectorizar la query una sola vez
        embeddings = await asyncio.to_thread(self.embedder.embed, [query])
        query_embedding = embeddings[0]

        # Construir filtros base (category, extension, etc.)
        should_conditions = []
        if filters:
            for key, value in filters.items():
                if not value:
                    continue
                if isinstance(value, list):
                    for v in value:
                        should_conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
                else:
                    should_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        must_conditions = []
        # Support for ExifTool Ranges (sizes, years)
        if range_filters:
            for field, bounds in range_filters.items():
                range_args = {}
                if "gte" in bounds and bounds["gte"] is not None:
                    range_args["gte"] = float(bounds["gte"])
                if "lte" in bounds and bounds["lte"] is not None:
                    range_args["lte"] = float(bounds["lte"])
                if range_args:
                    must_conditions.append(FieldCondition(key=field, range=Range(**range_args)))
                
        # Support for ExifTool Exact matches (Author, dynamic metadata)
        if exact_filters:
            for field, value in exact_filters.items():
                if value is not None:
                    must_conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))

        # ── Payload filtering por visibilidad (RBAC) ──────────────
        # Excluir documentos admin_only para cualquier rol que no sea admin.
        visibility_condition = (
            FieldCondition(key="visibility", match=MatchValue(value="admin_only"))
            if role != "admin" else None
        )

        def _build_filter(extra_must_text=None):
            """Construye el Filter de Qdrant con soporte para must_not de visibilidad."""
            m = list(must_conditions)
            if extra_must_text:
                m.append(extra_must_text)
            return Filter(
                should=should_conditions if should_conditions else None,
                must=m if m else None,
                must_not=[visibility_condition] if visibility_condition else None,
            ) if (should_conditions or m or visibility_condition) else None

        # Filtros para ambas búsquedas
        semantic_filter = _build_filter()
        lexical_filter = _build_filter(
            extra_must_text=FieldCondition(key="text", match=MatchText(text=query_text))
        )

        # Ejecutar ambas búsquedas en paralelo
        semantic_task = asyncio.to_thread(
            self.client.query_points,
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=semantic_filter,
            limit=top_k,
            with_payload=True,
        )
        lexical_task = asyncio.to_thread(
            self.client.query_points,
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=lexical_filter,
            limit=top_k,
            with_payload=True,
        )

        semantic_results, lexical_results = await asyncio.gather(
            semantic_task, lexical_task
        )

        # Fusionar resultados con boost léxico
        LEXICAL_BOOST = 1.15

        # Indexar resultados léxicos por texto para lookup rápido
        lexical_texts = set()
        for point in lexical_results.points:
            lexical_texts.add(point.payload.get("text", ""))

        seen_texts = set()
        merged = []

        # Procesar resultados semánticos (aplicar boost si también son léxicos)
        for point in semantic_results.points:
            text = point.payload.get("text", "")
            if text in seen_texts:
                continue
            seen_texts.add(text)

            score = point.score
            if text in lexical_texts:
                score = min(score * LEXICAL_BOOST, 1.0)  # Cap a 1.0

            merged.append({
                "text": text,
                "score": score,
                "source": point.payload.get("source", "unknown"),
                "category": point.payload.get("category", "General"),
                "extension": point.payload.get("extension", ""),
                "page": point.payload.get("page", None),
                "chunk_index": point.payload.get("chunk_index", None),
            })

        # Añadir resultados léxicos que no estaban en semánticos
        for point in lexical_results.points:
            text = point.payload.get("text", "")
            if text in seen_texts:
                continue
            seen_texts.add(text)

            merged.append({
                "text": text,
                "score": point.score * LEXICAL_BOOST,
                "source": point.payload.get("source", "unknown"),
                "category": point.payload.get("category", "General"),
                "extension": point.payload.get("extension", ""),
                "page": point.payload.get("page", None),
                "chunk_index": point.payload.get("chunk_index", None),
            })

        # Reordenar por score fusionado y truncar
        merged.sort(key=lambda r: r["score"], reverse=True)
        return merged[:top_k]

    async def get_by_source(self, source: str) -> list[dict]:
        """
        Recupera la totalidad de los fragmentos asociados a un archivo específico.

        Utiliza una operación de 'scroll' sobre Qdrant para obtener todos los 
        metadatos y textos sin necesidad de realizar una búsqueda por similitud. 
        Útil para reconstruir documentos para chat o visualización.
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
                "resumen": point.payload.get("resumen", None),
            })

        # Ordenar por chunk_index para reconstruir el documento en orden
        formatted.sort(key=lambda x: x.get("chunk_index", 0) or 0)
        return formatted

    async def update_document_summary(self, source: str, summary: str) -> None:
        """
        Persiste un resumen generado por IA en todos los fragmentos de un documento.

        Permite implementar una caché de resúmenes directamente en la base de 
        datos vectorial, evitando llamadas redundantes al LLM en el futuro.
        """
        await asyncio.to_thread(self.ensure_collection)
        await asyncio.to_thread(
            self.client.set_payload,
            collection_name=self.collection_name,
            payload={"resumen": summary},
            points=Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=source))
            ]),
        )
