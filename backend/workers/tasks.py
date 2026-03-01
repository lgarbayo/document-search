"""
workers/tasks.py — Orquestación del Pipeline de Ingesta (Celery Tasks).

Este módulo define la lógica de ejecución pesada que se procesa en segundo plano 
para evitar bloquear la API principal. Implementa una pipeline de 5 etapas para 
transformar documentos crudos en conocimiento indexado y buscable.

Etapas del Pipeline:
    1. Extracción: Obtiene texto y metadatos (PDF, Office, Imágenes mediante OCR).
    2. Limpieza: Normaliza el texto eliminando ruidos y artefactos de extracción.
    3. Resumen (Best-effort): Genera una síntesis con IA para pre-visualización rápida.
    4. Fragmentación (Chunking): Divide el texto preservando el contexto semántico.
    5. Indexación: Vectoriza los fragmentos y los persiste en la base de datos Qdrant.

Invocación:
    Disparado asíncronamente desde los endpoints de carga de archivos.
"""

import logging
from workers.celery_app import celery_app
from services.document_extractor import (
    extract_document_content,
    clean_text,
    chunk_text,
    deduplicate_chunks,
)
from services.vector_db import VectorDBService
from core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="workers.tasks.process_document")
def process_document(self, file_path: str, original_filename: str) -> dict:
    """
    Ejecuta el ciclo de vida completo de ingesta de un documento.

    Utiliza `self.update_state` para reportar el progreso granular al frontend, 
    permitiendo que el usuario vea en qué etapa exacta (Extracción, Limpieza, etc.) 
    se encuentra su archivo.

    Args:
        self (Task): Referencia a la tarea para gestión de estados.
        file_path (str): Ruta al archivo temporal en el servidor.
        original_filename (str): Nombre del archivo para visualización en resultados.

    Returns:
        dict: Estadísticas finales del procesamiento (fragmentos generados, 
              duplicados eliminados, etc.).

    Raises:
        Exception: Cualquier error crítico abortará la tarea y se registrará en Redis.
    """
    try:
        # ── Paso 1: Extraer texto y metadatos ──
        self.update_state(state="PROCESSING", meta={"step": "extracting_text"})
        logger.info(f"📄 Extrayendo contenido de: {original_filename}")
        raw_text, doc_metadata = extract_document_content(file_path)
        logger.info(f"   → {len(raw_text)} caracteres extraídos. Categoría: {doc_metadata.get('category')}")

        # ── Paso 2: Limpiar ──
        self.update_state(state="PROCESSING", meta={"step": "cleaning_text"})
        logger.info("🧹 Limpiando texto...")
        cleaned_text = clean_text(raw_text)
        logger.info(f"   → {len(cleaned_text)} caracteres tras limpieza")

        # ── Paso 2b: Generar resumen con LLM (no bloqueante — best-effort) ──
        self.update_state(state="PROCESSING", meta={"step": "summarizing"})
        summary = ""
        try:
            from services.llm_service import get_llm_service
            llm = get_llm_service()
            summary = llm.summarize(cleaned_text[:5000])
            logger.info(f"📝 Resumen generado: {len(summary)} caracteres")
        except Exception as e:
            logger.warning(f"⚠️  Resumen LLM no disponible ({type(e).__name__}): {e}")

        # ── Paso 3: Fragmentar ──
        self.update_state(state="PROCESSING", meta={"step": "chunking"})
        logger.info(f"✂️  Fragmentando (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})...")
        chunks = chunk_text(cleaned_text, size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
        logger.info(f"   → {len(chunks)} chunks generados")

        # ── Paso 4: Deduplicar ──
        self.update_state(state="PROCESSING", meta={"step": "deduplicating"})
        total_before = len(chunks)
        chunks = deduplicate_chunks(chunks)
        duplicates_removed = total_before - len(chunks)
        logger.info(f"🔍 Deduplicación: {duplicates_removed} duplicados eliminados, {len(chunks)} chunks únicos")

        # ── Paso 5: Vectorizar y guardar ──
        self.update_state(state="PROCESSING", meta={"step": "embedding_and_storing"})
        logger.info("🧠 Vectorizando e insertando en Qdrant...")

        vdb = VectorDBService()
        vdb.ensure_collection()

        # Crear metadata para cada chunk combinando los globales del doc
        metadata = []
        for i in range(len(chunks)):
            chunk_meta = {
                "source": original_filename,
                "chunk_index": i,
                "summary": summary,
            }
            # Añadir metadatos deducidos (category, file_size, extension)
            chunk_meta.update(doc_metadata)
            metadata.append(chunk_meta)

        inserted = vdb.upsert(chunks=chunks, metadata=metadata)
        logger.info(f"✅ {inserted} vectores insertados en Qdrant")

        # ── Resultado ──
        result = {
            "status": "completed",
            "filename": original_filename,
            "total_chunks": total_before,
            "unique_chunks": len(chunks),
            "duplicates_removed": duplicates_removed,
            "characters_extracted": len(raw_text),
            "characters_cleaned": len(cleaned_text),
        }

        logger.info(f"🎉 Procesamiento completado: {original_filename}")
        return result

    except Exception as e:
        logger.error(f"❌ Error procesando {original_filename}: {e}")
        # Celery marca la tarea como FAILURE automáticamente si hay excepción
        raise
