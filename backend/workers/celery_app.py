"""
workers/celery_app.py — Configuración del Motor de Tareas Asíncronas (Celery).

Este módulo orquestara el procesamiento en segundo plano de MeigaSearch. 
Utiliza Redis como agente de mensajería (broker) para coordinar las tareas 
y como almacén de resultados para rastrear el estado de la indexación.

Configuración técnica:
    - Broker/Backend: Redis (URL configurada en settings).
    - Serialización: JSON (estándar para interoperabilidad).
    - Estabilidad: Reintentos automáticos y confirmación tardía (Acks Late).

Instrucciones de ejecución (Docker):
    celery -A workers.celery_app worker --loglevel=info
"""

from celery import Celery

from core.config import settings

# Instancia principal de Celery.
# "meiga_worker" identifica la aplicación dentro del ecosistema de workers.
celery_app = Celery(
    "meiga_worker",
    broker=settings.REDIS_URL,          # Cola donde se depositan las tareas.
    backend=settings.REDIS_URL,         # Donde se guardan los resultados/errores.
    include=["workers.tasks"],          # Registro de módulos que contienen @task.
)

# Afinamiento de la configuración para producción.
celery_app.conf.update(
    # Seguridad y formato de datos
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Localización temporal para logs y scheduling corregido
    timezone="Europe/Madrid",
    enable_utc=True,

    # Estrategia de Fiabilidad:
    # task_acks_late: El worker confirma la tarea SOLO después de terminarla. 
    # Si el worker muere a mitad, la tarea vuelve a la cola (evita pérdida de datos).
    task_acks_late=True,
    
    # worker_prefetch_multiplier: Solo reserva 1 tarea por worker simultáneamente. 
    # Crítico cuando las tareas consumen mucha RAM (como la extracción de PDFs grandes).
    worker_prefetch_multiplier=1,

    # Ciclo de vida de los resultados (1 hora) para no saturar Redis.
    result_expires=3600,

    # Compatibilidad con versiones modernas de Celery
    broker_connection_retry_on_startup=True,
)
