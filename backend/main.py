"""
main.py — Punto de entrada de la aplicación FastAPI.

Este módulo inicializa la aplicación FastAPI, configura los middlewares,
monta los archivos estáticos para el frontend y gestiona el ciclo de vida
de la aplicación (eventos de inicio y cierre).

Responsabilidades principales:
    - Inicialización de la app FastAPI con metadatos.
    - Configuración de CORS para peticiones entre dominios.
    - Gestión del ciclo de vida (pre-carga de servicios).
    - Integración del router de la API.
    - Servicio de archivos estáticos para la interfaz web.

Uso:
    Ejecutar el servidor usando uvicorn:
    $ uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from services.vector_db import VectorDBService

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona los eventos del ciclo de vida de la aplicación.
    
    Inicio (Startup):
        - Inicializa el servicio VectorDBService.
        - Pre-carga el proveedor de embeddings y los modelos locales.
        - Esto evita que la primera petición API sufra una latencia alta
          debido a la carga perezosa (lazy loading) de modelos pesados de ML.
    
    Cierre (Shutdown):
        - Espacio para tareas de limpieza (ej. cerrar conexiones a DB).
    """
    # Pre-carga del servicio de Vector DB y el proveedor de embeddings al arrancar.
    # Este es un paso crítico de "pre-calentamiento" para proyectos OSS que usan ML,
    # ya que asegura que el sistema esté listo para servir peticiones de inmediato.
    _ = VectorDBService()
    yield
    # La lógica de apagado se puede añadir aquí si es necesario.

app = FastAPI(
    title="Meiga — Asistente de Conocimiento Corporativo",
    description=(
        "Backend avanzado de RAG (Generación Aumentada por Recuperación). "
        "Soporta ingesta asíncrona de documentos (PDF, Office, Imágenes), "
        "búsqueda semántica híbrida y chat RAG global con citaciones."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Ubicación de Swagger UI
    redoc_url="/redoc",    # Ubicación de ReDoc
)

# Configuración de CORS (Cross-Origin Resource Sharing).
# Esencial para permitir que el frontend (potencialmente en otro puerto o dominio)
# se comunique con esta API de forma segura.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos por seguridad.
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP estándar (GET, POST, etc.).
    allow_headers=["*"],  # Permitir cabeceras estándar.
)

# Registro del router principal de la API.
# Todos los endpoints de búsqueda, subida y gestión tienen el prefijo /api.
app.include_router(router, prefix="/api")


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Endpoint de comprobación de estado (Health Check).
    Útil para healthchecks de Docker y herramientas de monitorización (ej. Kubernetes).
    """
    return {
        "status": "ok", 
        "service": "meiga-backend",
        "version": app.version
    }


# ── Servicio de Archivos Estáticos del Frontend ───────────────────
# En un entorno de contenedores (Docker), el frontend suele ser compilado
# y colocado en directorios específicos. El backend sirve estos archivos
# para permitir un despliegue de un solo endpoint.

FRONTEND_PATH = "/app/frontend/index.html"
FRONTEND_ASSETS_PATH = "/app/frontend/assets"

# Montar activos estáticos (CSS, JS, Imágenes).
# StaticFiles es eficiente para servir contenido que no cambia dinámicamente.
if os.path.isdir(FRONTEND_ASSETS_PATH):
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_PATH), name="assets")

@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """
    Sirve el punto de entrada principal (index.html) de la aplicación frontend.
    Esto permite que el mismo servidor aloje tanto la API como la interfaz de usuario.
    """
    if os.path.isfile(FRONTEND_PATH):
        return FileResponse(FRONTEND_PATH, media_type="text/html")
    
    # Respuesta informativa si faltan los archivos del frontend.
    return {
        "error": "Frontend no encontrado.",
        "hint": "Asegúrate de que el directorio frontend/ esté correctamente montado en el contenedor Docker."
    }
