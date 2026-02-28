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

        # Reverse-map: los chips del frontend envían tipos como "pdf", "contract", "invoice"
        # pero Qdrant almacena la extensión real del archivo (.pdf, .csv, etc.)
        type_to_extensions = {
            "pdf": [".pdf", ".txt", ".png", ".jpg", ".jpeg"],  # Documentos generales
            "contract": [".pdf"],   # Contratos son PDFs
            "invoice": [".csv", ".xlsx"],  # Facturas son datos tabulares
            "code_snippet": [".txt"],  # Código fuente
            "proposal": [".pdf"],  # Propuestas son PDFs
        }

        filters = {}
        if type:
            # Convertir tipos del frontend a extensiones reales
            extensions = []
            for t in type:
                extensions.extend(type_to_extensions.get(t, []))
            # Deduplicar
            extensions = list(set(extensions))
            if extensions:
                filters["extension"] = extensions

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
        raw_results = await vdb.search(query=q_normalized, top_k=top_k, filters=filters if filters else None)

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
# ═══════════════════════════════════════════════════════════════
#  /api/document — Vista detallada de un documento indexado
# ═══════════════════════════════════════════════════════════════

from fastapi.responses import FileResponse

@router.get("/document/view", tags=["Documentos"])
async def view_document(source: str = Query(..., description="Ruta del archivo fuente")):
    """
    Sirve el archivo original directamente para visualizarlo (PDF, Imagen, etc.).
    """
    file_path = Path(source)
    
    # Fallback para desarrollo local: si la BD (en Docker) guardó '/app/datasets/docs/archivo.pdf'
    # pero FastAPI se está ejecutando en el Mac, buscar en '../datasets/docs/archivo.pdf'
    if not file_path.exists() and str(file_path).startswith("/app/datasets/"):
        relative_path = str(file_path)[len("/app/datasets/"):]
        local_path = Path("../datasets") / relative_path
        if local_path.exists():
            file_path = local_path

    if not file_path.exists() or not file_path.is_file():
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

        # Obtener tamaño del archivo si existe (incluyendo fallback local)
        file_path = Path(source)
        if not file_path.exists() and str(file_path).startswith("/app/datasets/"):
            relative_path = str(file_path)[len("/app/datasets/"):]
            local_path = Path("../datasets") / relative_path
            if local_path.exists():
                file_path = local_path
                
        file_size = file_path.stat().st_size if file_path.exists() else None

        # Metadatos
        ext = file_path.suffix.lower()
        type_map = {
            ".pdf": "pdf", ".txt": "pdf", ".csv": "invoice",
            ".xlsx": "invoice", ".png": "pdf", ".jpg": "pdf", ".jpeg": "pdf",
        }

        return {
            "source": source,
            "title": file_path.stem.replace("_", " ").title(),
            "extension": ext,
            "type": type_map.get(ext, "pdf"),
            "category": chunks[0].get("category", "General") if chunks else "General",
            "totalChunks": len(chunks),
            "wordCount": len(full_text.split()),
            "fileSize": file_size,
            "chunks": [{"text": c["text"], "chunkIndex": c.get("chunk_index"), "page": c.get("page")} for c in chunks],
            "fullText": full_text,
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
