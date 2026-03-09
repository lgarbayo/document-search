"""
workers/tasks.py — LA CADENA DE MONTAJE (Tareas de Celery).
---------------------------------------------------------
Aquí es donde ocurre el "trabajo pesado". Imagina una cadena de montaje 
de una fábrica. Un archivo entra por un lado y sale convertido en 
"Conocimiento Vectorial" por el otro.

PASOS DE LA FÁBRICA:
1. EXTRACCIÓN: Abrimos el archivo y leemos qué dice (incluso si es una foto).
2. LIMPIEZA: Quitamos las "manchas" o basura del texto.
3. RESUMEN: La IA lee rápido y nos dice de qué trata (opcional).
4. FRAGMENTACIÓN: Cortamos el texto en trozos pequeños (chunks).
5. INDEXACIÓN: Guardamos esos trozos en el "Cerebro" (Qdrant) para poder buscarlos.
"""

import logging

from services.document_extractor import (
    extract_document_content,
    clean_text,
    chunk_text,
    deduplicate_chunks,
)
from services.vector_db import VectorDBService
from core.config import settings

logger = logging.getLogger(__name__)


def process_document(file_path: str, original_filename: str) -> dict:
    """
    Ejecuta el ciclo de vida completo de ingesta de un documento.

    Utiliza `self.update_state` para reportar el progreso granular al frontend, 
    permitiendo que el usuario vea en qué etapa exacta (Extracción, Limpieza, etc.) 
    se encuentra su archivo.

    Args:
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
        logger.info(f"📄 Extrayendo contenido de: {original_filename}")
        raw_text, doc_metadata = extract_document_content(file_path)
        logger.info(f"   → {len(raw_text)} caracteres extraídos. Categoría: {doc_metadata.get('category')}")

        # ── Paso 2: Limpiar ──
        logger.info("🧹 Limpiando texto...")
        cleaned_text = clean_text(raw_text)
        logger.info(f"   → {len(cleaned_text)} caracteres tras limpieza")

        # ── Paso 2b: Generar resumen con LLM (no bloqueante — best-effort) ──
        summary = ""
        try:
            from services.llm_service import get_llm_service
            llm = get_llm_service()
            summary = llm.summarize(cleaned_text[:5000])
            logger.info(f"📝 Resumen generado: {len(summary)} caracteres")
        except Exception as e:
            logger.warning(f"⚠️  Resumen LLM no disponible ({type(e).__name__}): {e}")

        # ── Paso 3: Fragmentar ──
        logger.info(f"✂️  Fragmentando (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})...")
        chunks = chunk_text(cleaned_text, size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
        logger.info(f"   → {len(chunks)} chunks generados")

        # ── Paso 4: Deduplicar ──
        total_before = len(chunks)
        chunks = deduplicate_chunks(chunks)
        duplicates_removed = total_before - len(chunks)
        logger.info(f"🔍 Deduplicación: {duplicates_removed} duplicados eliminados, {len(chunks)} chunks únicos")

        # ── Paso 5: Vectorizar y guardar ──
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
