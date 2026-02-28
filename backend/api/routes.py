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
from services.document_extractor import SUPPORTED_EXTENSIONS
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
    top_k: int = Query(5, ge=1, le=20, description="Número de resultados"),
    type: list[str] = Query(None, description="Filtrar por tipo de documento"),
):
    """
    Búsqueda semántica sobre los documentos indexados.

    Devuelve respuesta en el formato esperado por el frontend Angular:
    SearchResponse { results, total, page, pageSize, durationMs, queryEchoed }
    """
    try:
        start_time = time.time()

        # Construir filtros
        filters = {}
        if type:
            filters["extension"] = type

        vdb = VectorDBService()
        raw_results = await vdb.search(query=q, top_k=top_k)

        # Transformar al formato SearchResponse del frontend
        results = []
        for r in raw_results:
            source = r.get("source", "unknown")
            ext = Path(source).suffix.lower().lstrip(".")  # "pdf", "csv", etc.

            # Mapear extensión a DocumentType del frontend
            type_map = {
                "pdf": "pdf",
                "txt": "pdf",        # El frontend no tiene tipo "txt"
                "csv": "invoice",    # Mapeo razonable para datos tabulares
                "xlsx": "invoice",
                "png": "pdf",        # Imágenes procesadas por OCR
                "jpg": "pdf",
                "jpeg": "pdf",
            }
            doc_type = type_map.get(ext, "pdf")

            # Construir DocumentMetadata
            now_iso = datetime.now(timezone.utc).isoformat()
            document = {
                "id": str(uuid.uuid4()),
                "title": Path(source).stem.replace("_", " ").title(),
                "type": doc_type,
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

        return {
            "results": results,
            "total": len(results),
            "page": 1,
            "pageSize": top_k,
            "durationMs": duration_ms,
            "queryEchoed": q,
        }

    except Exception as e:
        logger.error(f"❌ Error en búsqueda: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error realizando la búsqueda: {e}"
        )


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
    Busca ocurrencias de la query en el texto para resaltado.
    Devuelve lista de {start, end} con las posiciones.
    """
    highlights = []
    query_lower = query.lower()
    text_lower = text.lower()
    start = 0

    while True:
        pos = text_lower.find(query_lower, start)
        if pos == -1:
            break
        highlights.append({"start": pos, "end": pos + len(query)})
        start = pos + 1

    return highlights
