"""
Embeds every chunk in chunks.jsonl and stores it in a local Chroma vector DB.

Uses Chroma's built-in default embedding function (a local ONNX build of
all-MiniLM-L6-v2, via onnxruntime) instead of sentence-transformers/torch --
free, runs on CPU, no API key needed, and much lighter on memory than the
full torch runtime, which matters since api.py loads this same collection
on a memory-capped hosting tier.

Each chunk is stored as TWO vectors that return the same text:
  - the plain chunk text, and
  - the chunk text prefixed with its page title and heading breadcrumb.
A chunk like "Divyakant Gupta / Founder & CEO / ..." never mentions
"leadership", so the plain vector misses questions like "who founded
IDCUBE?"; the contextual one ("... About IDCUBE > Leadership > ...") catches
them. But contextual-only hurts other queries (every title says "IDCUBE",
so everything looks alike), so both are kept and ask.retrieve() dedupes.

Chroma persists to disk (--persist-dir), so this only needs to be re-run
when chunks.jsonl changes (i.e. after a weekly re-scrape + re-chunk).

Usage:
  python build_index.py --chunks chunks.jsonl --persist-dir chroma_db
"""

import argparse
import json
import shutil
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

COLLECTION_NAME = "idcube_india"
BATCH_SIZE = 64
CONTEXT_ID_SUFFIX = "#ctx"


def contextual_text(chunk: dict) -> str:
    return f"{chunk.get('title') or ''}\n{chunk.get('heading_path') or ''}\n{chunk['text']}"


def main():
    ap = argparse.ArgumentParser(description="Embed chunks and build the Chroma index")
    ap.add_argument("--chunks", default="chunks.jsonl")
    ap.add_argument("--persist-dir", default="chroma_db")
    args = ap.parse_args()

    chunks = [json.loads(line) for line in open(args.chunks, encoding="utf-8")]
    print(f"Loaded {len(chunks)} chunks")

    embed = DefaultEmbeddingFunction()
    # Wipe and rebuild rather than delete_collection(): Chroma leaves the old
    # collection's segment folder on disk, so stale copies of the index pile
    # up and get shipped with every deploy.
    if Path(args.persist_dir).exists():
        shutil.rmtree(args.persist_dir)
    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.create_collection(COLLECTION_NAME, metadata={"unique_chunks": len(chunks)})

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        metadatas = [
            {
                "chunk_id": c["chunk_id"],
                "url": c["url"],
                "title": c.get("title") or "",
                "heading_path": c.get("heading_path") or "",
                "source_type": c["source_type"],
            }
            for c in batch
        ]
        collection.add(
            ids=[c["chunk_id"] for c in batch] + [c["chunk_id"] + CONTEXT_ID_SUFFIX for c in batch],
            embeddings=list(embed(texts)) + list(embed([contextual_text(c) for c in batch])),
            documents=texts + texts,
            metadatas=metadatas + metadatas,
        )
        print(f"  embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"Done. Index has {len(chunks)} chunks ({collection.count()} vectors), persisted at {args.persist_dir}/")


if __name__ == "__main__":
    main()
