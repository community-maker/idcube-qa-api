"""
Refuse to publish an index that shrank sharply compared with the live one.

A mass drop in chunks almost always means the crawl failed (e.g. Cloudflare
challenged the crawler, or a sitemap didn't load), not that the website
deleted most of its content. Exiting non-zero fails the CI run -- so GitHub
emails the repo owner -- instead of silently shipping a near-empty index.

Usage:
  python pipeline/check_index_size.py --chunks chunks.jsonl --current chroma_db
"""

import argparse
import sys
from pathlib import Path

MIN_RATIO = 0.8
COLLECTION_NAME = "idcube_india"


def current_chunk_count(persist_dir: str) -> int:
    if not Path(persist_dir).exists():
        return 0
    import chromadb

    try:
        collection = chromadb.PersistentClient(path=persist_dir).get_collection(COLLECTION_NAME)
    except Exception as e:  # first run, or an unreadable old index
        print(f"No readable current index ({e}); skipping the size check")
        return 0
    return (collection.metadata or {}).get("unique_chunks") or collection.count()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--current", required=True)
    args = ap.parse_args()

    new = sum(1 for _ in open(args.chunks, encoding="utf-8"))
    old = current_chunk_count(args.current)
    print(f"New index: {new} chunks; live index: {old} chunks")
    if old and new < MIN_RATIO * old:
        sys.exit(
            f"Refusing to publish: {new} chunks is under {MIN_RATIO:.0%} of the live {old}. "
            "The crawl probably failed -- check the scrape step's log."
        )


if __name__ == "__main__":
    main()
