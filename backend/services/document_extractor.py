"""
services/document_extractor.py — Pipeline de procesamiento de documentos.

Soporta múltiples formatos:
  - PDF  → PyMuPDF (fitz)
  - TXT  → lectura directa
  - CSV  → lectura como texto tabular
  - XLSX → openpyxl, convierte filas a texto

Pipeline:
  extract_document_content() → clean_text() → chunk_text() → deduplicate_chunks()
"""

import csv
import hashlib
import io
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
import logging

try:
    from pdf2image import convert_from_path
    from PIL import Image, ImageOps
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

# Extensiones soportadas
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx", ".png", ".jpg", ".jpeg"}


def extract_document_content(file_path: str) -> tuple[str, dict]:
    """
    Extrae texto y metadatos básicos de un archivo.

    Formatos soportados: PDF, TXT, CSV, XLSX.

    Args:
        file_path: Ruta absoluta al archivo.

    Returns:
        Tupla (texto_crudo, metadatos_dict).
        Metadatos incluye: file_size, category, extension.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato no es soportado o no hay texto.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Formato no soportado: {ext}. "
            f"Formatos válidos: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Dispatch según extensión
    extractors = {
        ".pdf": _extract_pdf,
        ".txt": _extract_txt,
        ".csv": _extract_csv,
        ".xlsx": _extract_xlsx,
        ".png": _extract_img,
        ".jpg": _extract_img,
        ".jpeg": _extract_img,
    }

    text, extra_meta = extractors[ext](file_path)

    if not text.strip():
        raise ValueError(f"El archivo no contiene texto extraíble: {file_path}")

    # Extraer metadatos
    file_size = os.path.getsize(file_path)
    category = _infer_category(text, ext, filename=path.stem)

    metadata = {
        "extension": ext,
        "file_size_bytes": file_size,
        "category": category,
    }
    
    # Extraer metadatos técnicos avanzados con ExifTool
    try:
        import subprocess
        import json
        result = subprocess.run(
            ["exiftool", "-j", str(path)],
            capture_output=True,
            text=True,
            check=False
        )
        if result.stdout:
            parsed_exif = json.loads(result.stdout)
            if isinstance(parsed_exif, list) and len(parsed_exif) > 0:
                exif_data = parsed_exif[0]
                for k in ["Directory", "SourceFile", "FileName", "ExifToolVersion"]:
                    exif_data.pop(k, None)
                metadata["exif_metadata"] = exif_data
                
                # Intentar extraer el año (YYYY) de CreateDate o FileModifyDate para filtrado numérico en Qdrant
                date_str = exif_data.get("CreateDate") or exif_data.get("FileModifyDate")
                if date_str and isinstance(date_str, str) and len(date_str) >= 4:
                    try:
                        metadata["exif_year"] = int(date_str[:4])
                    except ValueError:
                        pass
    except Exception as e:
        logger.warning(f"No se pudo extraer metadata ExifTool para {file_path}: {e}")
    
    # Combinar con los metadatos específicos del extractor (ej. autor de PDF)
    if extra_meta:
        metadata.update(extra_meta)

    return text, metadata


def _infer_category(text: str, ext: str, filename: str = "") -> str:
    """
    Infiere la categoría corporativa usando heurísticas avanzadas.
    Combina análisis del contenido + nombre del archivo con pesos diferenciados.
    """
    text_lower = text[:8000].lower()
    filename_lower = filename.lower() if filename else ""

    categories = {
        "RRHH": [
            "vacaciones", "nómina", "nóminas", "contrato laboral", "empleado",
            "salario", "despido", "baja laboral", "ausencia", "beneficios",
            "recursos humanos", "selección", "candidato", "onboarding",
            "teletrabajo", "jornada", "convenio", "antigüedad", "plantilla",
            "offer letter", "puesto", "retribución", "compensación",
        ],
        "Finanzas": [
            "factura", "facturación", "iva", "balance", "gastos",
            "presupuesto", "ingresos", "contabilidad", "fiscal", "pago",
            "pedido", "proveedor", "cobro", "deuda", "amortización",
            "cuenta", "financiero", "trimestre", "resultado", "margen",
            "ventas", "revenue", "coste", "precio", "descuento", "euro",
        ],
        "Legal": [
            "acuerdo", "confidencialidad", "ley", "decreto", "demanda",
            "litigio", "cláusula", "poder notarial", "notario", "nda",
            "licitación", "pliego", "condiciones", "adjudicación",
            "constitución", "estatutos", "escritura", "mercantil",
            "propiedad intelectual", "rgpd", "protección de datos",
        ],
        "Técnico": [
            "api", "servidor", "base de datos", "código", "despliegue",
            "arquitectura", "software", "hardware", "firmware",
            "iot", "sensor", "panel", "manual", "especificación",
            "ficha técnica", "configuración", "instalación", "versión",
        ],
        "Comercial": [
            "propuesta", "oferta", "cliente", "comercial", "catálogo",
            "producto", "servicio", "proyecto", "licitación", "smart",
            "solución", "demo", "piloto", "partnership", "colaboración",
        ],
        "Sostenibilidad": [
            "sostenibilidad", "medioambiental", "emisiones", "carbono",
            "residuos", "reciclaje", "ods", "rsc", "impacto ambiental",
            "huella", "certificación ambiental", "iso 14001", "energía",
        ],
        "IT/Soporte": [
            "incidencia", "soporte", "ticket", "resolución", "sla",
            "equipo", "inventario", "activo", "licencia", "renovación",
            "backup", "antivirus", "red", "vpn", "usuario",
        ],
    }

    scores = {cat: 0 for cat in categories}

    for cat, keywords in categories.items():
        for kw in keywords:
            # Peso 1× por cada aparición en el texto
            scores[cat] += text_lower.count(kw)
            # Peso 3× si la keyword aparece en el nombre del archivo
            if kw in filename_lower:
                scores[cat] += 3

    # Extra: si es CSV/XLSX sin coincidencias claras, sesgo a Finanzas o IT/Soporte
    if ext in [".csv", ".xlsx"]:
        if scores["Finanzas"] == 0 and scores["IT/Soporte"] == 0:
            scores["Finanzas"] += 1

    best_category = max(scores, key=scores.get)

    if scores[best_category] == 0:
        return "General"

    return best_category


def _extract_pdf(file_path: str) -> tuple[str, dict]:
    """Extrae texto de un PDF usando PyMuPDF. Si falla o no encuentra texto útil, usa OCR."""
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"No se pudo abrir el PDF: {e}")

    # Extraer metadatos nativos del PDF
    pdf_meta = {}
    if doc.metadata:
        for key in ["author", "creator", "title", "subject", "keywords", "producer"]:
            val = doc.metadata.get(key)
            if val and isinstance(val, str) and val.strip():
                # Normalizar nombres de metadatos (ej., creator -> empresa/software)
                pdf_meta[key.lower()] = val.strip()

    text_parts = []
    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            text_parts.append(page_text)

    doc.close()
    extracted_text = "\n".join(text_parts)

    # Heurística: Si el PDF tiene menos de 50 caracteres (ej. una imagen escaneada gigante),
    # intentamos usar Tesseract OCR como fallback.
    if len(extracted_text.strip()) < 50 and OCR_AVAILABLE:
        logger.info(f"Poco texto extraído de {file_path}. Iniciando Tesseract OCR fallback...")
        try:
            # Convertimos las páginas del PDF a imágenes (máximo 300 DPI)
            images = convert_from_path(file_path, dpi=300)
            ocr_text_parts = []
            for img in images:
                # Corregir rotación EXIF si la hay
                img = ImageOps.exif_transpose(img)
                # Extraemos texto usando los modelos en español e inglés
                page_ocr = pytesseract.image_to_string(img, lang="spa+eng", config="--psm 1")
                ocr_text_parts.append(page_ocr)
            
            extracted_text = "\n".join(ocr_text_parts)
            logger.info("OCR finalizado exitosamente.")
        except Exception as e:
            logger.warning(f"Fallo en el OCR fallback para {file_path}: {e}")

    return extracted_text, pdf_meta


def _extract_img(file_path: str) -> tuple[str, dict]:
    """Extrae texto de una imagen nativa (PNG, JPG) usando Tesseract OCR."""
    if not OCR_AVAILABLE:
        raise ValueError("Tesseract OCR no está instalado. No se pueden procesar imágenes puras.")
    
    try:
        logger.info(f"Procesando imagen con OCR: {file_path}")
        img = Image.open(file_path)
        # Corregir rotación EXIF típica de móviles
        img = ImageOps.exif_transpose(img)
        
        # OSD (Orientation and Script Detection) forzar auto-rotación si no hay EXIF (--psm 1)
        extracted_text = pytesseract.image_to_string(img, lang="spa+eng", config="--psm 1")
        return extracted_text, {}
    except Exception as e:
        raise ValueError(f"Fallo al realizar OCR sobre la imagen: {e}")


def _extract_txt(file_path: str) -> tuple[str, dict]:
    """Lee un archivo de texto plano."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(), {}


def _extract_csv(file_path: str) -> tuple[str, dict]:
    """
    Lee un CSV y lo convierte en texto legible.
    Cada fila se convierte en: "columna1: valor1 | columna2: valor2 | ..."
    """
    lines = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convierte cada fila en texto clave:valor
            parts = [f"{key}: {value}" for key, value in row.items() if value]
            lines.append(" | ".join(parts))

    return "\n".join(lines), {}


def _extract_xlsx(file_path: str) -> tuple[str, dict]:
    """
    Lee un XLSX y lo convierte en texto legible.
    Usa los headers como claves, igual que CSV.
    """
    try:
        from openpyxl import load_workbook
    except ImportError:
        raise ImportError(
            "openpyxl es necesario para leer archivos XLSX. "
            "Instala con: pip install openpyxl"
        )

    wb = load_workbook(file_path, read_only=True, data_only=True)
    all_text = []

    for sheet in wb.sheetnames:
        ws = wb[sheet]
        rows = list(ws.iter_rows(values_only=True))

        if not rows:
            continue

        # Primera fila como headers
        headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows[0])]

        for row in rows[1:]:
            parts = []
            for header, value in zip(headers, row):
                if value is not None:
                    parts.append(f"{header}: {value}")
            if parts:
                all_text.append(" | ".join(parts))

    wb.close()
    return "\n".join(all_text), {}


def clean_text(text: str) -> str:
    """
    Limpia el texto crudo extraído.

    Operaciones:
      - Normaliza espacios en blanco
      - Elimina líneas que son solo números (pies de página)
      - Colapsa saltos de línea excesivos
      - Strip de cada línea
    """
    text = re.sub(r'[^\S\n\t]+', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
    return text.strip()


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Fragmenta el texto en chunks con solapamiento, respetando límites de oración.

    Intenta cortar en finales de oración para preservar el contexto semántico.
    Si no encuentra un final de oración, corta en el último espacio.

    Args:
        text: Texto limpio.
        size: Tamaño máximo de cada chunk (caracteres).
        overlap: Solapamiento entre chunks.

    Returns:
        Lista de chunks.
    """
    if not text:
        return []

    # Separadores de oración (en orden de preferencia)
    sentence_endings = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*')

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + size

        if end < text_length:
            # Buscar el último final de oración dentro del rango
            segment = text[start:end]
            best_break = -1

            for match in sentence_endings.finditer(segment):
                # Solo considerar quiebres que dejen un chunk razonable (>30% del tamaño)
                if match.start() > size * 0.3:
                    best_break = match.end()

            if best_break > 0:
                end = start + best_break
            else:
                # Fallback: cortar en el último espacio
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < text_length else text_length

    return chunks


STOP_WORDS_ES_EN = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "en", "por", "con", "para", "sin",
    "sobre", "entre", "hacia", "desde", "como", "que", "se",
    "su", "sus", "lo", "es", "son", "no", "si", "ya", "más",
    "muy", "pero", "este", "esta", "estos", "estas", "ese",
    "esa", "esos", "esas", "todo", "toda", "todos", "todas",
    "otro", "otra", "otros", "otras",
    "the", "of", "and", "in", "to", "for", "is", "on",
    "with", "at", "by", "an", "or", "not", "are", "was",
    "be", "has", "had", "its", "from", "this", "that", "which", "also",
}


def normalize_query(query: str) -> str:
    """
    Normaliza una query de búsqueda para mejorar la precisión.

    Operaciones:
      - Strip de espacios
      - Colapsar espacios múltiples
      - Eliminar puntuación huérfana al inicio/final
      - Eliminar stop words (artículos, preposiciones)
    """
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)
    query = re.sub(r'^[^\w]+|[^\w]+$', '', query)
    # Filtrar stop words para mejorar la búsqueda vectorial/léxica
    words = [w for w in query.split() if w.lower() not in STOP_WORDS_ES_EN]
    return ' '.join(words) if words else query  # fallback si todo son stop words


def deduplicate_chunks(chunks: list[str]) -> list[str]:
    """
    Elimina chunks duplicados (SHA256). Preserva orden original.
    """
    seen: set[str] = set()
    unique: list[str] = []

    for chunk in chunks:
        h = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)

    return unique
