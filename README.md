# MeigaSearch

**Un buscador corporativo basado en RAG y búsqueda semántica local, sin dependencias en APIs externas.**

![Status](https://img.shields.io/badge/status-Hackathon%202026-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

## 🎯 Propósito

MeigaSearch resuelve el problema de búsqueda a través de documentos corporativos dispersos (PDFs, Excel, Word, PPTX, CSV, texto, imágenes). Utiliza modelos de IA locales para indexación y búsqueda semántica, manteniendo la soberanía de los datos sin depender de servicios en la nube.

## ✨ Características Principales

- **Ingesta Mágica**: Sube documentos y se indexan automáticamente con OCR avanzado (Tesseract) y extracción de metadatos técnicos (ExifTool).
- **Global RAG Chat**: Chatea con toda tu base de conocimientos mediante **Streaming (SSE)** y obtén respuestas con **citas precisas** a los documentos originales.
- **Búsqueda Híbrida Inteligente**: Combina búsqueda semántica (vectores) + búsqueda léxica (exacta) en paralelo para obtener resultados óptimos.
- **Inferencia de Categorías**: Clasificación automática del contenido en áreas como RRHH, Finanzas, Legal, Técnico, Comercial, Sostenibilidad e IT.
- **Provider-Agnostic LLM**: Soporte dinámico para proveedores locales (Qwen/SmolLM) y externos (OpenAI, Gemini, Claude) configurable en caliente.
- **Filtros Avanzados**: Búsqueda por autor, mes, año, creador del documento y metadatos técnicos específicos.
- **Autenticación JWT**: Control de acceso por roles (admin, editor, normal) con cifrado `bcrypt`.
- **Arquitectura Robusta**: Qdrant como base de datos vectorial y Celery + Redis para procesamiento asíncrono.

## � Formatos Soportados

| Tipo | Extensiones |
| :--- | :--- |
| **Documentos** | `.pdf`, `.docx`, `.pptx`, `.txt`, `.md`, `.html` |
| **Datos** | `.csv`, `.xlsx`, `.json`, `.xml` |
| **Imágenes** | `.png`, `.jpg`, `.jpeg` |

## 🚀 Instalación Rápida

### Con Docker Compose (Recomendado)

1. **Configura el entorno:**
   ```bash
   cd meiga-search/backend
   cp .env.example .env  # Configura tus API Keys si usas modelos externos
   ```

2. **Levanta los servicios:**
   ```bash
   docker compose up -d
   ```

3. **Accede a la interfaz:**
   - **Frontend**: [http://localhost:8000](http://localhost:8000)
   - **API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Credenciales por defecto

| Usuario | Password | Rol |
| :--- | :--- | :--- |
| `admin` | `admin123` | Administrador |
| `empleado` | `normal123` | Normal |
| `lector` | `lector123` | Lector |

## ⚙️ Configuración del LLM

Puedes cambiar el proveedor de inteligencia artificial en tiempo real desde el panel de administración o mediante la API:

```bash
curl -X POST http://localhost:8000/api/system/settings \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "gemini",
    "api_key": "YOUR_GEMINI_KEY",
    "model_name": "gemini-1.5-pro"
  }'
```

Proveedores soportados: `local`, `openai`, `gemini`, `claude`.

## 📖 Uso de la API Search

**Endpoint:** `GET /api/search`

**Parámetros:**
-   `q`: Consulta de búsqueda (ej. "factura de electricidad")
-   `mode`: `semantic` (default) o `text` (exacto)
-   `month` / `year`: Filtrado cronológico
-   `type`: `pdf`, `txt`, `csv`, `xlsx`, `image`
-   `author`: Filtrar por autor extraído de metadatos

## 🏗️ Arquitectura

```mermaid
graph TD
    subgraph Frontend ["💻 Frontend (SPA)"]
        UI["Vanilla JS + CSS Glassmorphism<br/><b>Marked.js</b> (Markdown)"]
    end

    subgraph Backend ["⚙️ API & Inteligencia"]
        FAST["<b>FastAPI</b> (Python 3.11+)<br/>JWT + Pydantic v2"]
        AI["<b>AI Engine:</b> LLMs & Embeddings<br/>(OpenAI, Gemini, Local)"]
    end

    subgraph Async ["⚡ Procesamiento & Ingesta"]
        WORK["<b>Celery + Redis</b><br/>(Pipeline asíncrono)"]
        EXT["<b>Extractores:</b> OCR (Tesseract)<br/>ExifTool + PyMuPDF"]
    end

    VDB[("🔍 Vector DB<br/><b>Qdrant</b>")]

    UI <--> FAST
    FAST <--> AI
    FAST <--> VDB
    FAST --> WORK
    WORK --> EXT
    EXT --> VDB

    %% Estilos de alta legibilidad (Texto oscuro sobre fondo claro)
    style Frontend fill:#e1f5fe,stroke:#01579b,color:#000
    style Backend fill:#fff3e0,stroke:#e65100,color:#000
    style Async fill:#f3e5f5,stroke:#4a148c,color:#000
    style VDB fill:#e8f5e9,stroke:#1b5e20,color:#000
    
    style UI fill:#fff,stroke:#01579b,color:#000
    style FAST fill:#fff,stroke:#e65100,color:#000
    style AI fill:#fff,stroke:#e65100,color:#000
    style WORK fill:#fff,stroke:#4a148c,color:#000
    style EXT fill:#fff,stroke:#4a148c,color:#000
```

> [!NOTE]
> **¿Qué es Marked.js?**
> [Marked.js](https://marked.js.org/) es un compilador de Markdown de alto rendimiento. En MeigaSearch, lo utilizamos en el frontend para transformar las respuestas de la IA (que vienen en formato Markdown) en HTML limpio y formateado (negritas, listas, bloques de código), garantizando una experiencia de lectura premium en el chat.

## 🧪 Testing

```bash
cd meiga-search/backend
pytest tests/
```

## 📄 Licencia

MeigaSearch se distribuye bajo la licencia **Apache 2.0**. Ver [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

---
Desarrollado durante el HackUDC 2026. Profesionalmente adaptado para entornos corporativos seguros.
