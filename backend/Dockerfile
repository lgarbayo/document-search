FROM python:3.14-slim

# Evita que Python genere archivos .pyc y activa logs inmediatos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependencias del sistema para PyMuPDF y ExifTool
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl tesseract-ocr tesseract-ocr-spa libimage-exiftool-perl && \
    rm -rf /var/lib/apt/lists/*

# Instala dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente
COPY . .

# Crea directorio de uploads
RUN mkdir -p /app/uploads

EXPOSE 8000
