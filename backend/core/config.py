"""
core/config.py — Configuración Centralizada del Proyecto.

Este módulo gestiona los ajustes de la aplicación usando pydantic-settings,
permitiendo la carga fluida de la configuración desde variables de entorno
o un archivo .env. Este patrón es estándar en proyectos OSS ya que
separa el código de datos sensibles o específicos del entorno.

Acceso a los ajustes a través de la instancia singleton:
    from core.config import settings
    print(settings.REDIS_URL)
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Ajustes de toda la aplicación y variables de entorno.
    
    Pydantic valida automáticamente los tipos y proporciona valores por defecto.
    En producción, sobrescriba estos valores configurando las variables de entorno
    correspondientes (ej., export OPENAI_API_KEY=sk-...).
    """

    # ─── Configuración de Redis ───
    # Utilizado como broker de mensajes y backend de resultados para Celery.
    # Por defecto apunta al servicio 'redis' definido en docker-compose.
    REDIS_URL: str = "redis://redis:6379/0"

    # ─── Configuración de Qdrant (Base de Datos Vectorial) ───
    # Host y puerto para el motor de almacenamiento vectorial Qdrant.
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "corporate_docs"

    # ─── Configuración del Modelo de Embeddings ───
    # Proveedor: "sentence-transformers" (local/gratuito) o "openai" (nube/pago).
    EMBEDDING_PROVIDER: str = "sentence-transformers"
    
    # Nombre del modelo utilizado por el proveedor.
    # Por defecto local: paraphrase-multilingual-MiniLM-L12-v2 (excelente para documentos multi-idioma).
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Dimensión de los Embeddings: Debe coincidir con la salida del modelo (384 para MiniLM).
    EMBEDDING_DIM: int = 384

    # ─── Claves de API (Proveedores LLM) ───
    # Dejar cadenas vacías por defecto; sobrescribir vía .env o variables de entorno.
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # ─── Configuración de LLM (Modelo de Lenguaje) ───
    # Proveedor principal para Chat RAG y Resúmenes.
    # Opciones: "local" (solo CPU), "openai", "gemini", "claude".
    LLM_PROVIDER: str = "local"
    
    # Identificadores específicos de modelo para cada proveedor en la nube.
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
    GEMINI_LLM_MODEL: str = "gemini-2.5-flash"
    CLAUDE_LLM_MODEL: str = "claude-3-haiku-20240307"

    # ─── Parámetros de Procesamiento de Documentos ───
    # Límite de caracteres para cada fragmento para asegurar que quepan en el contexto del LLM.
    CHUNK_SIZE: int = 1000      
    
    # Solapamiento entre fragmentos para preservar el contexto en los límites.
    CHUNK_OVERLAP: int = 200    
    
    # Directorio donde se persisten temporalmente los documentos subidos.
    UPLOAD_DIR: str = "/app/uploads"

    class Config:
        """Configuración de Pydantic para la clase Settings."""
        # Cargar variables desde .env si está presente.
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        # Ignorar variables en el entorno que no estén definidas aquí.
        # Esto evita que la aplicación falle debido a variables de entorno irrelevantes.
        extra = "ignore"


# Instancia singleton compartida en toda la aplicación.
# Esto asegura una única fuente de verdad para todas las configuraciones.
settings = Settings()
