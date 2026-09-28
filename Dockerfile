FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake Chroma's ONNX embedding model into the image at build time, so
# containers don't need to download it on every cold start (Render's free
# tier spins the container down after inactivity and starts fresh each time).
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up'])"

COPY ask.py api.py ./
COPY chroma_db ./chroma_db

ENV CHROMA_DIR=/app/chroma_db
ENV PYTHONUNBUFFERED=1

# Render injects $PORT at runtime; default to 8000 for local/manual runs.
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
