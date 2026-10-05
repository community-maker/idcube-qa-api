"""
Splits crawled IDCUBE page/PDF JSON into small chunks for embedding.

Approach: split each document on Markdown headings first (so a chunk never
straddles two unrelated sections, e.g. "Mercury MP1501 specs" won't get
mixed with "Mercury MP1502 specs"), then further split any section still
longer than ~180 words into paragraph-sized pieces with a little overlap so
a fact sitting right at a paragraph boundary doesn't get orphaned.

Usage:
  python chunker.py --data-dir data --out chunks.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
TARGET_WORDS = 180
OVERLAP_WORDS = 30


def split_by_headings(markdown: str):
    """Yield (heading_path, section_text) chunks split at Markdown headings."""
    matches = list(HEADING_RE.finditer(markdown))
    if not matches:
        yield ("", markdown)
        return

    if matches[0].start() > 0:
        yield ("", markdown[: matches[0].start()])

    # Stack of (level, text). Pop by level, not by list position: pages often
    # skip levels (H1 -> H3), and slicing by position then treated the
    # previous H3 sibling as the parent of the next one.
    heading_stack = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading_text = m.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text))

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()

        heading_path = " > ".join(text for _, text in heading_stack)
        yield (heading_path, (heading_text + "\n" + body) if body else heading_text)


def explode_paragraph(para: str):
    """Break a single over-long paragraph into <=TARGET_WORDS pieces, on line
    boundaries where possible. PDF text comes out as one "paragraph" per PDF
    page (often 300-700 words), which the paragraph splitter alone can't cut."""
    if len(para.split()) <= TARGET_WORDS:
        return [para]
    pieces, current, words = [], [], 0
    for line in para.split("\n"):
        line_words = line.split()
        if len(line_words) > TARGET_WORDS:
            if current:
                pieces.append("\n".join(current))
                current, words = [], 0
            for j in range(0, len(line_words), TARGET_WORDS):
                pieces.append(" ".join(line_words[j : j + TARGET_WORDS]))
            continue
        if words + len(line_words) > TARGET_WORDS and current:
            pieces.append("\n".join(current))
            current, words = [], 0
        current.append(line)
        words += len(line_words)
    if current:
        pieces.append("\n".join(current))
    return pieces


def split_long_section(text: str):
    """Paragraph-based split with word-count overlap for long sections."""
    paragraphs = [
        piece
        for p in text.split("\n\n") if p.strip()
        for piece in explode_paragraph(p.strip())
    ]
    chunks, current, current_words = [], [], 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > TARGET_WORDS and current:
            chunks.append("\n\n".join(current))
            # Overlap by words, not whole paragraphs: carrying the last full
            # paragraph over doubled chunk size whenever paragraphs were big.
            tail = "\n\n".join(current).split()[-OVERLAP_WORDS:]
            current, current_words = [" ".join(tail)], len(tail)
        current.append(para)
        current_words += para_words

    if current:
        chunks.append("\n\n".join(current))
    return chunks if chunks else [text]


PDF_PAGE_HEADER_RE = re.compile(r"^www\.idcubesystems\.com\s+Page \d+ of \d+$", re.IGNORECASE)


def repair_shifted_font(text: str) -> str:
    """Some PDFs (e.g. the Mercury installation guides) embed a font whose
    character codes are all shifted by a constant: "MP1501" extracts as
    "ˀ˃ʤʨʣʤ" (each char +627). Detect it -- most of the text in the
    U+0100-U+03FF block, where real text here never lives -- and shift back,
    taking the offset from the most common scrambled char, which is 'e'."""
    visible = [c for c in text if not c.isspace() and c != "\x03"]
    shifted = [c for c in visible if 0x100 <= ord(c) <= 0x3FF]
    if not visible or len(shifted) < 0.3 * len(visible):
        return text
    offset = ord(Counter(shifted).most_common(1)[0][0]) - ord("e")
    out = []
    for c in text:
        o = ord(c)
        if c == "\x03":  # these fonts encode the space as glyph 3
            out.append(" ")
        elif o >= 0x100 and 0x20 <= o - offset <= 0x7E:
            out.append(chr(o - offset))
        else:
            out.append(c)
    return "".join(out)


def repair_letter_spacing(line: str) -> str:
    """'H o w  A c c e s s 3 6 0  H e l i x' -> 'How Access360 Helix':
    single spaces between letters, two or more between words."""
    tokens = [t for t in line.split(" ") if t]
    if len(tokens) < 6 or sum(len(t) == 1 for t in tokens) < 0.6 * len(tokens):
        return line
    return " ".join(seg.replace(" ", "") for seg in re.split(r" {2,}", line.strip()) if seg.strip())


def is_junk_line(line: str) -> bool:
    """Decorative icon-font glyphs ('ʱ̅ʇ͔ǻͧ', stray 'ۯ') come out of some
    brochures as lines of noise. Keep a line only if at least half its
    characters are ASCII or Devanagari/Arabic script."""
    visible = [c for c in line if not c.isspace()]
    if len(visible) <= 2:
        # Short lines are often datasheet table values ("12", "4K") -- keep
        # those; drop only stray symbols/glyphs.
        return not any(c.isascii() and c.isalnum() for c in visible)
    readable = sum(ord(c) < 0x80 or 0x0600 <= ord(c) <= 0x06FF or 0x0900 <= ord(c) <= 0x097F
                   for c in visible)
    return readable < 0.5 * len(visible)


def clean_pdf_text(text: str) -> str:
    """pypdf output is padded with runs of spaces, repeats the site's
    'www.idcubesystems.com Page N of M' header on every page, and for some
    PDFs comes out letter-spaced or in a shifted font (repaired here)."""
    text = repair_shifted_font(text)
    lines = []
    for line in text.splitlines():
        line = repair_letter_spacing(line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if PDF_PAGE_HEADER_RE.match(line):
            continue
        if line and is_junk_line(line):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def title_from_url(url: str) -> str:
    """'.../BASF-case-study-1.pdf' -> 'BASF case study'."""
    stem = Path(urlparse(url).path).stem
    stem = re.sub(r"[-_ ]\d{1,2}$", "", stem)
    return re.sub(r"[-_]+", " ", stem).strip()


def chunk_document(doc: dict, source_type: str):
    # PDF records carry plain "text" only (no markdown) -- without this
    # fallback every PDF was silently skipped.
    markdown = doc.get("markdown") or clean_pdf_text(doc.get("text", ""))
    if len(markdown.strip()) < 20:
        return []
    title = doc.get("title") or title_from_url(doc["url"])

    chunks = []
    for heading_path, section_text in split_by_headings(markdown):
        section_text = section_text.strip()
        if not section_text:
            continue
        if len(section_text.split()) <= TARGET_WORDS * 1.4:
            pieces = [section_text]
        else:
            pieces = split_long_section(section_text)

        for piece in pieces:
            if len(piece.split()) < 8:  # skip near-empty fragments
                continue
            chunks.append({
                "url": doc["url"],
                "title": title,
                "source_type": source_type,
                "heading_path": heading_path,
                "text": piece.strip(),
                "word_count": len(piece.split()),
            })
    return chunks


def main():
    ap = argparse.ArgumentParser(description="Chunk crawled IDCUBE JSON for embedding")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="chunks.jsonl")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    all_chunks = []

    for source_type, subdir in (("page", "pages"), ("pdf", "pdfs")):
        folder = data_dir / subdir
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*.json")):
            doc = json.loads(f.read_text(encoding="utf-8"))
            all_chunks.extend(chunk_document(doc, source_type))

    # The site publishes the same pages under global, /in/, /us/ and /mea/
    # paths, so identical paragraphs appear up to 4 times -- enough to fill a
    # top-5 search with copies of one passage. Keep the first occurrence.
    seen_text = set()
    deduped = []
    for c in all_chunks:
        key = " ".join(c["text"].lower().split())
        if key in seen_text:
            continue
        seen_text.add(key)
        deduped.append(c)
    print(f"Dropped {len(all_chunks) - len(deduped)} duplicate chunks")
    all_chunks = deduped

    for i, c in enumerate(all_chunks):
        c["chunk_id"] = f"{c['source_type']}-{i:05d}"

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    total_words = sum(c["word_count"] for c in all_chunks)
    print(f"Wrote {len(all_chunks)} chunks ({total_words} words total) to {out_path}")


if __name__ == "__main__":
    main()
