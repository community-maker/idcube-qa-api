"""
The IDCUBE website assistant's server: the retrieval index plus the chat
relay that answers visitors with AIsha's LangChain agent (see chat_proxy.py
and aisha_agent.py).

/search does retrieval ONLY: question -> embed -> search Chroma -> return the
matching chunks. The agent's search_website tool uses the same retrieval.

Run:
  pip install -r requirements.txt
  uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /search   {"question": "...", "top_k": 8}  ->  {"results": [...]}
  GET  /health
  /chat/*        website chat with AIsha (see chat_proxy.py)
  GET  /widget.js      the embeddable chat widget
  GET  /widget-demo    a bare page with the widget, for testing before embedding
"""

import logging
import os
from pathlib import Path

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from ask import COLLECTION_NAME, TOP_K, retrieve
from chat_proxy import ALLOWED_ORIGINS, router as chat_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="IDCUBE Search API", version="1.0")
# Browsers may only call the API from the IDCUBE site itself.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(chat_router)

WIDGET_PATH = Path(__file__).with_name("widget.js")


@app.on_event("startup")
def load_index():
    chroma_client = chromadb.PersistentClient(path=os.environ.get("CHROMA_DIR", "chroma_db"))
    app.state.collection = chroma_client.get_collection(COLLECTION_NAME)
    # Load the embedding model now, not on the first visitor's search.
    retrieve("warm up", app.state.collection)


class SearchRequest(BaseModel):
    question: str
    top_k: int = TOP_K


class SearchResult(BaseModel):
    text: str
    url: str
    title: str = ""
    heading_path: str = ""


class SearchResponse(BaseModel):
    results: list[SearchResult]


@app.get("/health")
def health():
    collection = getattr(app.state, "collection", None)
    if collection is None:
        return {"status": "ok", "chunks_indexed": 0}
    unique = (collection.metadata or {}).get("unique_chunks", collection.count())
    return {"status": "ok", "chunks_indexed": unique}


@app.get("/widget.js", include_in_schema=False)
def widget():
    return FileResponse(
        WIDGET_PATH,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/widget-demo", include_in_schema=False)
def widget_demo():
    return HTMLResponse(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>IDCUBE chat widget demo</title></head>"
        "<body style='font-family:sans-serif;padding:40px'>"
        "<h1>IDCUBE chat widget demo</h1>"
        "<p>The chat button is in the bottom-right corner.</p>"
        "<script src='/widget.js' data-api='' defer></script>"
        "</body></html>"
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    chunks = retrieve(req.question, app.state.collection, k=req.top_k)
    results = [
        SearchResult(
            text=c["text"],
            url=c["url"],
            title=c.get("title", ""),
            heading_path=c.get("heading_path", ""),
        )
        for c in chunks
    ]
    return SearchResponse(results=results)
