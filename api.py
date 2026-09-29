"""
REST API wrapper around the IDCUBE retrieval index, for an external agent
platform (Purple Fabric) to use as a "search this website's data" tool.

This service does retrieval ONLY: question -> embed -> search Chroma ->
return the matching chunks. It does not call any LLM itself -- the calling
agent platform (which has its own LLM step) is expected to take these chunks
as context and generate the final answer.

Run:
  pip install fastapi uvicorn
  uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /search   {"question": "...", "top_k": 8}  ->  {"results": [...]}
  GET  /health
  /chat/*        website chat relay to Purple Fabric (see chat_proxy.py)
  GET  /widget.js      the embeddable chat widget
  GET  /widget-demo    a bare page with the widget, for testing before embedding
"""

import os
from pathlib import Path

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from ask import COLLECTION_NAME, TOP_K, retrieve
from chat_proxy import ALLOWED_ORIGINS, router as chat_router

app = FastAPI(title="IDCUBE Search API", version="1.0")
# Browsers may only call /chat/* from the IDCUBE site itself. /search is
# called server-to-server by Purple Fabric, which CORS doesn't affect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(chat_router)

WIDGET_PATH = Path(__file__).with_name("widget.js")

_state = {}


@app.on_event("startup")
def load_index():
    chroma_client = chromadb.PersistentClient(path=os.environ.get("CHROMA_DIR", "chroma_db"))
    _state["collection"] = chroma_client.get_collection(COLLECTION_NAME)


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
    collection = _state.get("collection")
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
    chunks = retrieve(req.question, _state["collection"], k=req.top_k)
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
