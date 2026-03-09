"""
core/config.py — EL CENTRO DE MANDO (Configuración).
--------------------------------------------------
Este módulo es el responsable de centralizar todos los "interruptores" y llaves 
de la aplicación. En lugar de tener valores fijos por todo el código, los 
definimos aquí una sola vez.

¿POR QUÉ USAMOS PYDANTIC-SETTINGS?
1. SEGURIDAD: Nunca guardamos claves API reales en el código (commit).
2. FLEXIBILIDAD: Puedes cambiar cómo se comporta la app solo editando el archivo `.env`.
3. VALIDACIÓN: Si esperas un número y recibes una letra, la app te avisará al arrancar.

COMO USARLO EN EL CÓDIGO:
    from core.config import settings
    print(settings.LLM_PROVIDER) # Devuelve 'local', 'openai', etc.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    DEFINICIÓN DE VARIABLES (Design Tokens del Backend).
    
    Cada variable aquí definida representa una pieza del rompecabezas.
    Si una variable no está en el archivo .env, Pydantic usará el valor por defecto 
    que ves aquí escrito.
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
    GEMINI_LLM_MODEL: str = "gemini-2.0-flash"

    # ─── Parámetros de Procesamiento de Documentos ───
    # Límite de caracteres para cada fragmento para asegurar que quepan en el contexto del LLM.
    CHUNK_SIZE: int = 1000      
    
    # Solapamiento entre fragmentos para preservar el contexto en los límites.
    CHUNK_OVERLAP: int = 200    
    
    # Directorio donde se persisten temporalmente los documentos subidos.
    UPLOAD_DIR: str = "/app/uploads"

    class Config:
        """
        CONFIGURACIÓN DEL MOTOR DE AJUSTES.
        ----------------------------------
        Aquí le decimos a Pydantic: "Oye, busca primero en un archivo llamado .env".
        Esto nos permite tener configuraciones distintas en tu PC, en desarrollo y en producción.
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        # Ignorar variables externas que no hayamos definido arriba.
        extra = "ignore"


# Instancia singleton compartida en toda la aplicación.
# Esto asegura una única fuente de verdad para todas las configuraciones.
settings = Settings()
