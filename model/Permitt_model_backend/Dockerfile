FROM python:3.11-slim

WORKDIR /app

# Installiere Systemabhängigkeiten für PyTorch und andere Bibliotheken
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Kopiere requirements.txt zuerst, um Caching beim Erstellen des Images zu optimieren
COPY requirements.txt .

# Installiere Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den Projektcode
COPY . .

# Stelle sicher, dass die notwendigen Verzeichnisse existieren
RUN mkdir -p api/output/generated_images

# Port, auf dem die FastAPI-Anwendung läuft
EXPOSE 8000

# Starte die FastAPI-Anwendung mit Uvicorn
CMD ["uvicorn", "api.api_with_id:app", "--host", "0.0.0.0", "--port", "8000"]
