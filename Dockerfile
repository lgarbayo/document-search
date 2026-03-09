FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl tesseract-ocr tesseract-ocr-spa libimage-exiftool-perl && \
    rm -rf /var/lib/apt/lists/*

# Instala dependencias Python
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código del backend
COPY backend/ .

# Copia el frontend dentro de la imagen
COPY frontend/ /app/frontend/

# Crea directorio de uploads
RUN mkdir -p /app/uploads

EXPOSE 8000
