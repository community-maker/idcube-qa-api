FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch build first (the default PyPI wheel drags in ~9GB of CUDA
# packages we never use on a CPU-only host); once satisfied, the rest of
# requirements.txt installs against it instead of pulling the GPU build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY ask.py api.py ./
COPY chroma_db ./chroma_db

ENV CHROMA_DIR=/app/chroma_db

# Render injects $PORT at runtime; default to 8000 for local/manual runs.
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
