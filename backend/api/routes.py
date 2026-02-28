"""
api/routes.py — Endpoints de la API.

Tres endpoints:
  POST /api/upload       → Sube un PDF, dispara procesamiento async
  GET  /api/search       → Búsqueda semántica sobre los documentos
  GET  /api/status/{id}  → Estado de una tarea de procesamiento
"""

import os
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from workers.tasks import process_document
from services.vector_db import VectorDBService
from services.document_extractor import SUPPORTED_EXTENSIONS, normalize_query
from services.llm_expander import expand_query_async
from core.config import settings
from celery.result import AsyncResult

logger = logging.getLogger(__name__)

router = APIRouter()


# ═══════════════════════════════════════════════════════════════
#  POST /api/upload — Ingesta de documentos
# ═══════════════════════════════════════════════════════════════

@router.post("/upload", tags=["Ingesta"])
async def upload_document(file: UploadFile = File(...)):
    """
    Sube un documento para procesamiento asíncrono.

    Formatos soportados: PDF, TXT, CSV, XLSX.

    1. Valida la extensión del archivo
    2. Guarda el archivo en disco
    3. Dispara la tarea de Celery (NO bloquea)
    4. Devuelve HTTP 202 con el task_id para tracking
    """
    # Validar tipo de archivo
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: {ext}. Válidos: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Crear directorio de uploads si no existe
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Guardar archivo en disco
    file_path = upload_dir / file.filename
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"📁 Archivo guardado: {file_path} ({len(content)} bytes)")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error guardando el archivo: {e}"
        )

    # Disparar tarea de Celery (async, no bloquea FastAPI)
    task = process_document.delay(str(file_path), file.filename)

    logger.info(f"🚀 Tarea disparada: {task.id} para {file.filename}")

    return JSONResponse(
        status_code=202,
        content={
            "message": "Documento recibido. Procesamiento en curso.",
            "task_id": task.id,
            "filename": file.filename,
        }
    )


# ═══════════════════════════════════════════════════════════════
#  GET /api/status/{task_id} — Estado de la tarea
# ═══════════════════════════════════════════════════════════════

@router.get("/status/{task_id}", tags=["Ingesta"])
async def get_task_status(task_id: str):
    """
    Consulta el estado de una tarea de procesamiento.

    Estados posibles:
      - PENDING: La tarea está en la cola, esperando un worker.
      - PROCESSING: El worker está procesándola (con detalle del paso actual).
      - SUCCESS: Procesamiento completado exitosamente.
      - FAILURE: Error durante el procesamiento.

    Returns:
        Estado actual + metadata del resultado si completó.
    """
    result = AsyncResult(task_id)

    response = {
        "task_id": task_id,
        "status": result.status,
    }

    # Si está en proceso, incluir info del paso actual
    if result.state == "PROCESSING":
        response["detail"] = result.info

    # Si completó, incluir el resultado
    elif result.state == "SUCCESS":
        response["result"] = result.result

    # Si falló, incluir el error
    elif result.state == "FAILURE":
        response["error"] = str(result.result)

    return response


# ═══════════════════════════════════════════════════════════════
#  GET /api/search — Búsqueda semántica
# ═══════════════════════════════════════════════════════════════

@router.get("/search", tags=["Búsqueda"])
async def search_documents(
    q: str = Query(..., min_length=1, description="Texto de búsqueda"),
    top_k: int = Query(30, ge=1, le=1000, description="Número de resultados"),
    type: list[str] = Query(None, description="Filtrar por tipo de documento"),
    expand: bool = Query(False, description="Activar expansión de consulta con LLM"),
    min_size: int = Query(None, ge=0, description="Tamaño mínimo en bytes"),
    max_size: int = Query(None, ge=0, description="Tamaño máximo en bytes"),
    author: str = Query(None, description="Filtrar por autor del documento"),
    min_year: int = Query(None, ge=1900, description="Año mínimo de creación/modificación"),
    max_year: int = Query(None, ge=1900, description="Año máximo de creación/modificación"),
):
    """
    Búsqueda semántica sobre los documentos indexados.

    Modos de búsqueda:
      1. Normal: Búsqueda directa con la consulta del usuario
      2. Descriptiva (expand=true): Expande la consulta con LLM local (Qwen2.5-0.5B)
         para extraer palabras clave técnicas y mejorar resultados

    Fallback automático: Si la búsqueda normal no devuelve resultados,
    se activa automáticamente el modo descriptivo.

    Devuelve respuesta en el formato esperado por el frontend Angular:
    SearchResponse { results, total, page, pageSize, durationMs, queryEchoed }
    """
    try:
        start_time = time.time()

        filters = {}
        if type:
            extensions = []
            categories = []
            
            # Map known frontend file types to real extensions
            ext_map = {
                "pdf": [".pdf"],
                "txt": [".txt"],
                "csv": [".csv"],
                "xlsx": [".xlsx"],
                "image": [".png", ".jpg", ".jpeg", ".gif"],
            }
            
            for t in type:
                if t in ext_map:
                    extensions.extend(ext_map[t])
                else:
                    # Añadimos la Categoría en todas sus variantes para hacer el filtro Case-Insensitive
                    # en Qdrant (que hace match exacto por defecto)
                    categories.append(t)
                    categories.append(t.lower())
                    categories.append(t.capitalize())
            
            if extensions:
                filters["extension"] = list(set(extensions))
            if categories:
                filters["category"] = list(set(categories))
                
        exact_filters = {}
        if author:
            # MatchThext is helpful, but we can do a simplified case insensitive exact or array match
            # For exact filter using MatchValue, it works better if the UI sends the exact string or we just pass the raw input.
            exact_filters["author"] = author

        range_filters = {}
        if min_size is not None or max_size is not None:
            range_filters["file_size_bytes"] = {}
            if min_size is not None:
                range_filters["file_size_bytes"]["gte"] = min_size
            if max_size is not None:
                range_filters["file_size_bytes"]["lte"] = max_size
                
        if min_year is not None or max_year is not None:
            range_filters["exif_year"] = {}
            if min_year is not None:
                range_filters["exif_year"]["gte"] = min_year
            if max_year is not None:
                range_filters["exif_year"]["lte"] = max_year

        # Normalizar la query
        q_normalized = normalize_query(q)
        if not q_normalized:
            return {
                "results": [],
                "total": 0,
                "page": 1,
                "pageSize": top_k,
                "durationMs": 0,
                "queryEchoed": q,
            }

        vdb = VectorDBService()
        
        # 🔍 MODO DE BÚSQUEDA POR PALABRAS CLAVE (KeyBERT)
        # Estrategia:
        #   1. Si expand=true → extraer keywords de la query y COMBINAR resultados
        #      (query original + keywords) para maximizar cobertura hasta top_k
        #   2. Si expand=false → búsqueda normal; fallback a keywords si no hay resultados
        #
        # Nota: KeyBERT extrae el núcleo semántico de la query (no genera términos nuevos),
        # por eso combinar ambas búsquedas es más efectivo que reemplazar una por otra.

        extracted_keywords = None
        use_expansion = expand

        # Búsqueda inicial con query normalizada
        raw_results = await vdb.search(
            query=q_normalized, 
            top_k=top_k, 
            filters=filters if filters else None, 
            range_filters=range_filters if range_filters else None,
            exact_filters=exact_filters if exact_filters else None
        )

        # Activar extracción automáticamente si no hay resultados
        if not raw_results and not use_expansion:
            logger.info(f"🤖 Sin resultados para '{q}', activando extracción de palabras clave...")
            use_expansion = True

        # Si se activó la extracción (manual o automática)
        if use_expansion:
            try:
                logger.info(f"🔑 Extrayendo palabras clave de: '{q}'")
                extracted_keywords = await expand_query_async(q, max_keywords=5)
                logger.info(f"✅ Palabras clave extraídas: '{extracted_keywords}'")

                # Buscar con las keywords extraídas
                keyword_results = await vdb.search(
                    query=extracted_keywords,
                    top_k=top_k,
                    filters=filters if filters else None,
                    range_filters=range_filters if range_filters else None,
                    exact_filters=exact_filters if exact_filters else None
                )

                # --- FIXED MERGE ---
                # Build a unified result pool, tagging each result with the
                # query string that produced it so highlights work correctly.
                seen_texts: set[str] = set()
                merged: list[dict] = []

                for r in raw_results:
                    txt = r.get("text", "")
                    if txt not in seen_texts:
                        r["_highlight_query"] = q_normalized   # original query
                        merged.append(r)
                        seen_texts.add(txt)

                for r in keyword_results:
                    txt = r.get("text", "")
                    if txt not in seen_texts:
                        r["_highlight_query"] = extracted_keywords  # keyword query
                        merged.append(r)
                        seen_texts.add(txt)

                # Re-rank: results found by BOTH queries should rank highest.
                # Give a small score boost to keyword-only results so they are
                # not buried under low-scoring original results.
                keyword_texts = {r.get("text", "") for r in keyword_results}
                original_texts = {r.get("text", "") for r in raw_results}

                for r in merged:
                    txt = r.get("text", "")
                    if txt in keyword_texts and txt in original_texts:
                        r["score"] = r.get("score", 0.0) * 1.15  # boost overlap
                    elif txt in keyword_texts:
                        r["score"] = r.get("score", 0.0) * 1.05  # slight boost for keyword-only

                        raw_results = sorted(merged, key=lambda r: r.get("score", 0.0), reverse=True)[:top_k]
                        logger.info(f"🔀 Resultados combinados: {len(raw_results)} (query + keywords)")
                    elif keyword_results:
                        # Solo teníamos keywords results (fallback)
                        raw_results = keyword_results
            except Exception as e:
                logger.error(f"❌ Error en extracción de palabras clave: {e}")
                # Si falla la extracción, continuar con resultados originales (vacíos o no)
                pass

        # Transformar al formato SearchResponse del frontend
        results = []
        for r in raw_results:
            source = r.get("source", "unknown")
            ext = Path(source).suffix.lower().lstrip(".")  # "pdf", "csv", etc.

            # Construir DocumentMetadata
            now_iso = datetime.now(timezone.utc).isoformat()
            document = {
                "id": str(uuid.uuid4()),
                "title": Path(source).stem.replace("_", " ").title(),
                "type": r.get("category", "General"),
                "source": source,
                "author": None,
                "createdAt": now_iso,
                "updatedAt": now_iso,
                "tags": [ext.upper()] if ext else [],
                "sizeBytes": None,
            }

            # Construir HighlightedFragment
            text = r.get("text", "")
            fragment = {
                "text": text,
                "highlights": _find_highlights(text, q),
                "pageNumber": r.get("page"),
            }

            results.append({
                "document": document,
                "fragment": fragment,
                "relevanceScore": r.get("score", 0.0),
            })

        duration_ms = int((time.time() - start_time) * 1000)

        response = {
            "results": results,
            "total": len(results),
            "page": 1,
            "pageSize": top_k,
            "durationMs": duration_ms,
            "queryEchoed": q,
        }
        
        # Agregar metadata sobre extracción de palabras clave (si se usó)
        if use_expansion:
            response["searchMode"] = "descriptive"
            response["expandedQuery"] = extracted_keywords or ""
        else:
            response["searchMode"] = "normal"

        return response

    except Exception as e:
        logger.error(f"❌ Error en búsqueda: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error realizando la búsqueda: {e}"
        )


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
#  /api/document — Vista detallada de un documento indexado
# ═══════════════════════════════════════════════════════════════

from fastapi.responses import FileResponse

def _find_file_on_disk(source: str) -> Path | None:
    """Intenta encontrar el archivo físico, ya sea absoluto, relativo o solo por nombre."""
    file_path = Path(source)
    if file_path.exists() and file_path.is_file():
        return file_path

    # Fallback 1: Docker absolute to local relative
    if str(file_path).startswith("/app/datasets/"):
        rel_path = str(file_path)[len("/app/datasets/"):]
        local_path = Path("../datasets") / rel_path
        if local_path.exists() and local_path.is_file():
            return local_path
            
    # Fallback 2: Es solo un nombre de archivo (o no se encontró la ruta absoluta).
    # Buscamos en las carpetas de datos conocidas.
    search_dirs = ["/app/datasets", "/app/uploads", "../datasets", "../uploads"]
    filename_to_find = file_path.name
    
    for d in search_dirs:
        dir_path = Path(d)
        if not dir_path.exists() or not dir_path.is_dir():
            continue
        for root, _, files in os.walk(str(dir_path)):
            if filename_to_find in files:
                found = Path(root) / filename_to_find
                return found

    return None

@router.get("/document/view", tags=["Documentos"])
async def view_document(source: str = Query(..., description="Ruta del archivo fuente")):
    """
    Sirve el archivo original directamente para visualizarlo (PDF, Imagen, etc.).
    """
    file_path = _find_file_on_disk(source)
    
    if not file_path:
        raise HTTPException(status_code=404, detail="El archivo original ya no existe en disco.")
        
    return FileResponse(path=file_path)

@router.get("/document", tags=["Documentos"])
async def get_document_detail(source: str = Query(..., description="Ruta del archivo fuente")):
    """
    Recupera todos los chunks de un documento y los reconstruye con metadatos.
    """
    try:
        vdb = VectorDBService()
        chunks = await vdb.get_by_source(source)

        if not chunks:
            raise HTTPException(status_code=404, detail=f"Documento no encontrado: {source}")

        # Reconstruir texto completo
        full_text = "\n\n".join([c["text"] for c in chunks])

        # Obtener tamaño del archivo si existe (usando el helper)
        file_path = _find_file_on_disk(source)
        file_size = file_path.stat().st_size if file_path else None

        # Metadatos
        ext = file_path.suffix.lower()
        ext_clean = ext.lstrip(".")
        type_map = {
            "pdf": "pdf", "txt": "txt", "csv": "csv",
            "xlsx": "xlsx", "png": "image", "jpg": "image", "jpeg": "image",
        }
        doc_type = type_map.get(ext_clean, ext_clean)
        first_chunk = chunks[0] if chunks else {}

        # Extraer metadatos ExifTool almacenados en Ingesta
        exif_data = first_chunk.get("exif_metadata")

        return {
            "source": source,
            "title": file_path.stem.replace("_", " ").title() if file_path else source,
            "extension": ext,
            "type": doc_type,
            "category": first_chunk.get("category", "General"),
            "totalChunks": len(chunks),
            "wordCount": len(full_text.split()),
            "fileSize": file_size,
            "author": first_chunk.get("author"),
            "creator": first_chunk.get("creator"),
            "subject": first_chunk.get("subject"),
            "keywords": first_chunk.get("keywords"),
            "producer": first_chunk.get("producer"),
            "chunks": [{"text": c["text"], "chunkIndex": c.get("chunk_index"), "page": c.get("page")} for c in chunks],
            "fullText": full_text,
            "exif_metadata": exif_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error obteniendo documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
#  /api/system/* — Gestión del sistema (frontend integration)
# ═══════════════════════════════════════════════════════════════

@router.get("/system/pick-directory", tags=["Sistema"])
async def pick_directory():
    """
    En modo Docker no hay picker nativo del SO.
    Devuelve la ruta por defecto y lista de directorios disponibles.
    """
    datasets_path = "/app/datasets"
    if not os.path.isdir(datasets_path):
        datasets_path = "/app/uploads"

    # Listar directorios disponibles dentro de /app
    available = []
    for name in ["datasets", "uploads"]:
        full = f"/app/{name}"
        if os.path.isdir(full):
            # Contar archivos soportados
            count = sum(1 for _, _, files in os.walk(full) for f in files
                       if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS)
            available.append(f"{full} ({count} archivos)")

    return {"path": datasets_path, "available": available}


@router.get("/system/list-directory", tags=["Sistema"])
async def list_directory(path: str = Query("/app", description="Ruta a listar")):
    """
    Lista el contenido de un directorio dentro del contenedor.
    Devuelve carpetas y archivos con metadatos básicos.
    """
    target = Path(path)
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"No es un directorio válido: {path}")

    items = []
    try:
        for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.name.startswith('.'):
                continue
            if entry.is_dir():
                # Contar archivos soportados dentro
                file_count = sum(1 for _, _, files in os.walk(str(entry))
                                for f in files if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS)
                items.append({
                    "name": entry.name,
                    "type": "directory",
                    "path": str(entry),
                    "fileCount": file_count,
                })
            elif entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                items.append({
                    "name": entry.name,
                    "type": "file",
                    "path": str(entry),
                    "size": entry.stat().st_size,
                    "extension": entry.suffix.lower(),
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Sin permisos para leer: {path}")

    # Obtener la ruta padre para navegación
    parent = str(target.parent) if str(target) != "/" else None

    return {"current": str(target), "parent": parent, "items": items}


from pydantic import BaseModel

class IndexDirectoryRequest(BaseModel):
    path: str


@router.post("/system/index-directory", tags=["Sistema"])
async def index_directory(request: IndexDirectoryRequest):
    """
    Escanea un directorio y dispara tareas de Celery para cada archivo soportado.
    """
    dir_path = Path(request.path)
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"El directorio no existe: {request.path}")

    dispatched = 0
    for root, _, files in os.walk(str(dir_path)):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, filename)
                process_document.delay(file_path, filename)
                dispatched += 1
                logger.info(f"🚀 Tarea disparada para: {filename}")

    logger.info(f"📂 Directorio indexado: {request.path} → {dispatched} archivos")
    return {"dispatched": dispatched, "path": request.path}


@router.delete("/system/database", tags=["Sistema"])
async def clear_database():
    """
    Purga completa de la colección Qdrant. Elimina y recrea vacía.
    """
    try:
        vdb = VectorDBService()
        # Eliminar colección si existe
        try:
            vdb.client.delete_collection(collection_name=vdb.collection_name)
            logger.info(f"🗑️  Colección '{vdb.collection_name}' eliminada")
        except Exception:
            logger.info(f"Colección '{vdb.collection_name}' no existía, creando nueva...")

        # Recrear vacía
        vdb.ensure_collection()
        logger.info(f"✅ Colección '{vdb.collection_name}' recreada vacía")

        return {"status": "ok", "message": "Base de datos purgada correctamente"}
    except Exception as e:
        logger.error(f"❌ Error purgando la base de datos: {e}")
        raise HTTPException(status_code=500, detail=f"Error purgando la BD: {e}")


def _find_highlights(text: str, query: str) -> list[dict]:
    """
    Busca ocurrencias de cada palabra de la query en el texto para resaltado.
    Divide la query en palabras y busca cada una individualmente.
    Filtra palabras muy cortas (< 3 chars) para evitar ruido.
    Devuelve lista de {start, end} con las posiciones.
    """
    highlights = []
    text_lower = text.lower()

    # Dividir query en palabras, filtrar muy cortas
    words = [w for w in query.lower().split() if len(w) >= 3]
    if not words:
        # Fallback: buscar la query completa
        words = [query.lower()] if query.strip() else []

    for word in words:
        start = 0
        while True:
            pos = text_lower.find(word, start)
            if pos == -1:
                break
            highlights.append({"start": pos, "end": pos + len(word)})
            start = pos + 1

    # Ordenar y eliminar solapamientos
    highlights.sort(key=lambda h: h["start"])
    merged = []
    for h in highlights:
        if merged and h["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], h["end"])
        else:
            merged.append(h)

    return merged
