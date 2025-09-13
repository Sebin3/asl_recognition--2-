# Usar una imagen base estable con Python 3.11
FROM python:3.11

# Evitar que Python escriba archivos .pyc
ENV PYTHONDONTWRITEBYTECODE 1
# No almacenar en búfer la salida estándar
ENV PYTHONUNBUFFERED 1
# Puerto que usará Render
ENV PORT=10000

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo el archivo de requisitos primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Crear directorio para datos
RUN mkdir -p /app/data

# Copiar el modelo
COPY asl_recognition/data/model.pkl /app/data/model.pkl

# Copiar el resto de la aplicación
COPY . .

# Puerto expuesto (Render usa el puerto definido en la variable de entorno PORT)
EXPOSE $PORT

# Crear usuario y directorio de datos como root
RUN useradd -r -s /bin/false appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Cambiar al usuario no root
USER appuser

# Verificar que podemos escribir en el directorio de datos
RUN touch /app/data/.test_write && rm /app/data/.test_write

# Comando para ejecutar la aplicación con el puerto correcto para Render
# Usando shell form para que la variable de entorno se expanda correctamente
CMD exec uvicorn api:app --host 0.0.0.0 --port $PORT