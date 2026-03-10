"""
main.py — EL DIRECTOR DE ORQUESTA (FastAPI).
--------------------------------------------
Este es el archivo principal que se ejecuta al arrancar el backend. 
Imagina que es el pegamento que une la API (las rutas), la Seguridad (Auth), 
la Inteligencia Artificial (Servicios) y el Frontend (UI).

FLUJO DE ARRANQUE:
1. Lee la configuración (config.py).
2. Ejecuta 'lifespan': Carga los modelos de IA "pesados" en memoria.
3. Abre las puertas (CORS): Permite que el navegador hable con este servidor.
4. Monta la UI: Sirve los archivos HTML/JS para que veas la web.
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
    EL CICLO DE VIDA (Startup & Shutdown).
    --------------------------------------
    ¿Por qué existe esto? Porque cargar una IA tarda unos segundos.
    Si esperamos a que el primer usuario pregunte algo, el sistema parecerá lento.
    
    SOLUCIÓN: Usamos el Startup para "pre-calentar" (warm-up) los motores:
    - Cargamos el VectorDB y los Embeddings ANTES de quedar listos para peticiones.
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

# MIDDLEWARE DE CORS (Control de Accesos).
# ----------------------------------------
# Por seguridad, los navegadores bloquean peticiones que vienen de sitios distintos.
# Aquí le decimos: "Está bien, confío en mi frontend, déjale hablar conmigo".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En PROD, pon aquí tu dominio real (ej. https://busqueda.miempresa.com)
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
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
FRONTEND_LOCALES_PATH = "/app/frontend/locales"

# Montar activos estáticos (CSS, JS, Imágenes).
# StaticFiles es eficiente para servir contenido que no cambia dinámicamente.
if os.path.isdir(FRONTEND_ASSETS_PATH):
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_PATH), name="assets")

if os.path.isdir(FRONTEND_LOCALES_PATH):
    app.mount("/locales", StaticFiles(directory=FRONTEND_LOCALES_PATH), name="locales")

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
