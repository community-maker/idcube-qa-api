"""
IDCUBE Systems website crawler for RAG ingestion.

Strategy (why this, instead of a blind link-crawler or a plain readability
extractor):
  1. Read the site's own XML sitemaps (recursively, since they're nested
     indexes) to get the full, authoritative list of real pages. Far more
     complete and reliable than following <a> links, and exactly what the
     site's robots.txt invites crawlers to do (it even publishes an llms.txt).
  2. Fetch each page with a normal browser User-Agent and polite rate
     limiting. The site is server-rendered (plain HTTP GET returns full
     content, no JS execution needed) and sits behind Cloudflare but does not
     challenge well-behaved requests, so no headless browser is required.
  3. Extract content with BeautifulSoup + markdownify rather than a generic
     "readability" extractor (tried trafilatura first: it threw away ~80% of
     the real page -- this site is WordPress+Elementor with badly broken
     semantics, e.g. the *entire* page including the real <footer> is nested
     inside one <header> tag, which defeats every tag-based heuristic). We
     only strip genuinely non-visible tags (script/style/svg/forms/etc.) and
     keep everything else, converted to Markdown so heading/list/table
     structure survives for chunking.
  4. Because step 3 keeps everything, every page also carries the full mega
     menu, region-selector, and modal boilerplate that's identical across the
     whole site. We remove that in a second pass: after crawling, any line
     of text that appears verbatim on a large fraction of *distinct* pages
     is boilerplate (nav/footer/modals) and gets stripped, since real content
     is page-specific by definition.
  5. Decode Cloudflare's email obfuscation (data-cfemail) so contact emails
     come through as real addresses instead of a "[email protected]" placeholder
     -- this matters for a support-facing agent that needs to quote them.
  6. Track a content hash per URL in manifest.json so weekly re-runs can tell
     you exactly which pages changed, without re-embedding everything.

Usage:
  python idcube_scraper.py --region in --out data
  python idcube_scraper.py --region all --out data --include-pdf
  python idcube_scraper.py --region in --out data --limit 20   # quick test run

Output layout:
  data/
    pages/<slug>.json      one file per page: url, title, description, markdown, text, hash, lastmod, scraped_at
    pdfs/<slug>.json        (if --include-pdf) extracted PDF text
    manifest.json           url -> {hash, lastmod, scraped_at, changed_since_last_run}
    crawl_log.txt
"""

import argparse
import hashlib
import io
import json
import re
import sys
import time
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import requests
import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_markdown

BASE = "https://www.idcubesystems.com"
HOMEPAGE_URL = BASE + "/in/en/"

SITEMAP_SETS = {
    "global": ["/sitemap.xml"],
    "in": ["/india_sitemap.xml"],
    "us": ["/us_sitemap.xml"],
    "mea": ["/mea_sitemap.xml"],
}
PDF_SITEMAP = "/sitemap-pdf.xml"

# Legal/policy pages are deliberately excluded from every sitemap (common
# practice) but are still real, important pages -- included here by fixed
# URL since sitemap discovery will never find them.
EXTRA_URLS = [
    BASE + "/privacy-policy/",
    BASE + "/disclaimer/",
    BASE + "/partners/",
    BASE + "/partners/technology-partners/",
]

# The FAQ page renders its ~160 Q&A pairs client-side from JS, so the plain
# HTML pipeline above would only capture "Loading FAQs..." -- but the same
# data also ships as an inline JSON blob in the page source (a WordPress
# wp_localize_script pattern), which we parse directly instead of running a
# headless browser.
FAQ_URL = BASE + "/faq/"
FAQ_CONFIG_RE = re.compile(r"var\s+faqConfig\s*=\s*(\{.*?\});", re.DOTALL)

# --- PDF documents: readable names + categories, for exact download links ----
# Visitors ask "send me the brochure / datasheet / case study"; the agent can
# only answer with an exact PDF link if that link survives scraping (pages'
# "Download Now" buttons were reduced to bare text) and is labelled with
# words a visitor would actually search for.

PDF_HREF_RE = re.compile(r"\.pdf($|[?#])", re.IGNORECASE)
GENERIC_LINK_TEXT = {
    "", "download", "download now", "view", "view data sheet", "view datasheet",
    "read more", "click here", "learn more", "know more", "here",
}
DOC_CATEGORIES = {
    "brochures": ("Brochures and company profile", BASE + "/"),
    "case_studies": ("Case studies", BASE + "/support/resource-center/case-studies/"),
    "integration": ("Integration and application notes", BASE + "/support/resource-center/application-notes/"),
    "controllers": ("Controller and enclosure datasheets", BASE + "/support/resource-center/datasheet/"),
    "credentials": ("Reader, card and credential datasheets", BASE + "/support/resource-center/datasheet/"),
    "guides": ("Installation guides", BASE + "/support/resource-center/datasheet/"),
    "other": ("Other documents", BASE + "/support/resource-center/datasheet/"),
}
# Filenames that say too little on their own; names checked against each
# PDF's contents. Unlisted PDFs fall back to the rules in document_label().
DOC_OVERRIDES = {
    "IDCUBE-Company-profile-2026": ("brochures", "IDCUBE company profile 2026"),
    "IDCUBE-USA-V4.2": ("brochures", "IDCUBE USA brochure"),
    "IDCUBE-USA-Version": ("brochures", "IDCUBE USA brochure"),  # IDCUBE-USA-Version-4.pdf
    "GreenID_brochure": ("brochures", "GreenID brochure"),
    "IDCUBE-AXS-2024": ("brochures", "AXS+ access control reader brochure 2024"),
    "IDCUBE-AXS-2.0": ("brochures", "AXS+ 2.0 access control reader brochure"),
    "Agriculture-Bank-of-Egypt": ("case_studies", "Agriculture Bank of Egypt case study"),
    "Buisness-park": ("brochures", "Business park solution brochure"),
    "IDCUBE-Mobile-Credentials": ("brochures", "Mobile credentials presentation"),
    "IDCUBE_RFID_Credentials-2024": ("credentials", "IDCUBE RFID credentials 2024"),
    "ASSA-Flex": ("credentials", "ASSA Flex long-range RFID tag datasheet"),
    "Invixium": ("integration", "QRify on Invixium face readers integration note"),
    "IDCUBE-x-HID-Amico": ("integration", "HID Amico biometric readers integration note"),
    "Hospitality-Access-Management-Solution-Integration-Application-Note":
        ("integration", "ASSA ABLOY Vingcard hospitality integration note"),
    "IDCUBE-Milestone-VMS-Integration-Application-Note": ("integration", "Milestone XProtect VMS integration note"),
    "IDCUBE-Milestone-Integration": ("integration", "Milestone XProtect VMS integration note"),
    "IDCUBE-Genetec-Integration-Note": ("integration", "Genetec Security Center integration note"),
}
WORD_CASE = {
    "hid": "HID", "iclass": "iCLASS", "seos": "Seos", "prox": "Prox", "proxcard": "ProxCard",
    "proxkey": "ProxKey", "microprox": "MicroProx", "mifare": "MIFARE", "desfire": "DESFire",
    "idcube": "IDCUBE", "pacs": "PACS", "se": "SE", "ii": "II", "iii": "III", "8k": "8K",
}
CONTROLLER_RE = re.compile(
    r"\b(icaero|aero|icmlp\d*|icmmp\d*|icmmr\d*|mr\d+\w*|mp\d+|mercury|controller|access box)\b"
)
CREDENTIAL_RE = re.compile(
    r"\b(iclass|prox|proxcard|proxkey|microprox|seos|mifare|desfire|keyfob|card|tag|credentials?|reader|axs|idlr)\b"
)
# Industry-named PDFs ("Data-Center.pdf", "Hospitality.pdf") are the
# per-industry solution brochures.
INDUSTRY_RE = re.compile(
    r"\b(data cent(er|re)|enterprise|hospitality|multi family housing|co ?working|education|"
    r"business park|gated community|residential)\b"
)


def is_idcube_pdf(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.hostname is not None \
        and parsed.hostname.endswith("idcubesystems.com") and parsed.path.lower().endswith(".pdf")


def document_label(url: str):
    """-> (category key, readable name), e.g. 'hid-seos-8k-keyfob-ds-en.pdf'
    -> ('credentials', 'HID Seos 8K Keyfob datasheet')."""
    base = Path(urllib.parse.urlparse(url).path).stem
    while re.search(r"[-_]\d{1,2}$", base):  # upload re-copies: -1, _0, _0-1
        base = re.sub(r"[-_]\d{1,2}$", "", base)
    if base in DOC_OVERRIDES:
        return DOC_OVERRIDES[base]
    # "datasheet"/"ds" are dropped here and re-appended at the end below, so
    # "X100-Controller-Datasheet-V1.7" and "X100-Controller-v1.7" get the same
    # name and the catalog keeps just one of them.
    words = [w for w in re.sub(r"[-_]+", " ", base).split()
             if w.lower() not in ("en", "ds", "datasheet", "compressed")]
    words = [WORD_CASE.get(w.lower(), w.capitalize() if w.islower() else w) for w in words]
    words = [w.upper() if re.fullmatch(r"v\d+(\.\d+)*", w, re.I) else w for w in words]
    name = " ".join(words)
    lower = name.lower()
    if "case study" in lower:
        return "case_studies", name
    if "integration" in lower or "application note" in lower:
        return "integration", name
    if "installation guide" in lower:
        return "guides", name
    if "brochure" in lower:
        return "brochures", name
    for key, pattern in (("controllers", CONTROLLER_RE), ("credentials", CREDENTIAL_RE)):
        if pattern.search(lower):
            return key, name if "comparison" in lower else f"{name} datasheet"
    if INDUSTRY_RE.search(lower):
        return "brochures", f"{name} solution brochure"
    return "other", name


def inline_pdf_links(soup: BeautifulSoup, page_url: str, pdf_sink=None) -> None:
    """Replace each <a href="...pdf"> with plain text carrying the exact URL,
    since markdown conversion strips links down to their (usually generic)
    text. Collects the URLs into pdf_sink so linked-but-unlisted PDFs get
    crawled too -- the PDF sitemap doesn't list every PDF the site links to."""
    for a in soup.find_all("a", href=PDF_HREF_RE):
        url = urllib.parse.urldefrag(urllib.parse.urljoin(page_url, a["href"].strip()))[0]
        if not is_idcube_pdf(url):
            continue
        if pdf_sink is not None:
            pdf_sink.add(url)
        label = document_label(url)[1]
        text = a.get_text(" ", strip=True)
        shown = f"{label} (PDF): {url}"
        if text.lower() not in GENERIC_LINK_TEXT and text.lower() != label.lower():
            shown = f"{text} - {shown}"
        a.replace_with(soup.new_string(f" {shown} "))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

XML_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def write_json(path: Path, data):
    """Write JSON, working around Windows' 260-char MAX_PATH limit by using
    the extended-length path prefix when needed (common when the project
    lives several folders deep, e.g. under a synced/virtualized AppData path)."""
    text = json.dumps(data, ensure_ascii=False, indent=2)
    target = path
    if sys.platform == "win32":
        abs_str = str(path.resolve())
        if not abs_str.startswith("\\\\?\\"):
            abs_str = "\\\\?\\" + abs_str
        target = abs_str
    with open(target, "w", encoding="utf-8") as f:
        f.write(text)


def log(msg: str, log_file=None):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def fetch(url: str, session: requests.Session, retries: int = 3, timeout: int = 20):
    """GET with retry/backoff. Returns Response or None."""
    delay = 2
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 503):
                time.sleep(delay)
                delay *= 2
                continue
            return resp  # other codes: caller decides
        except requests.RequestException:
            if attempt == retries:
                return None
            time.sleep(delay)
            delay *= 2
    return None


def resolve_sitemap(url: str, session: requests.Session, seen: set, log_file=None):
    """Recursively resolve a sitemap or sitemap-index URL into leaf page URLs
    with their <lastmod>. Returns list of (url, lastmod)."""
    if url in seen:
        return []
    seen.add(url)

    resp = fetch(url, session)
    if resp is None or resp.status_code != 200:
        log(f"  ! failed to fetch sitemap {url} (status={getattr(resp, 'status_code', None)})", log_file)
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError:
        log(f"  ! could not parse XML at {url}", log_file)
        return []

    tag = root.tag.lower()
    results = []

    if tag.endswith("sitemapindex"):
        children = [
            el.find("sm:loc", XML_NS).text.strip()
            for el in root.findall("sm:sitemap", XML_NS)
            if el.find("sm:loc", XML_NS) is not None
        ]
        log(f"  sitemap index {url} -> {len(children)} child sitemaps", log_file)
        for child in children:
            results.extend(resolve_sitemap(child, session, seen, log_file))
    elif tag.endswith("urlset"):
        for el in root.findall("sm:url", XML_NS):
            loc_el = el.find("sm:loc", XML_NS)
            if loc_el is None or not loc_el.text:
                continue
            lastmod_el = el.find("sm:lastmod", XML_NS)
            lastmod = lastmod_el.text.strip() if lastmod_el is not None else None
            results.append((loc_el.text.strip(), lastmod))
    return results


def slugify(url: str) -> str:
    """Short, filesystem-safe, collision-resistant filename stem.
    Kept short (<=60 chars + 8-char hash) to stay well under Windows' MAX_PATH
    when the output dir itself is already deeply nested."""
    path = urllib.parse.urlparse(url).path.strip("/")
    slug = path.replace("/", "__") or "home"
    slug = unicodedata.normalize("NFKD", slug).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9_.-]", "-", slug)
    short_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:80]}-{short_hash}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


NON_VISIBLE_TAGS = [
    "script", "style", "noscript", "svg", "link", "input", "iframe",
    "form", "path", "button", "select", "option", "template",
]

# Cloudflare's email-obfuscation marker: <a class="__cf_email__" data-cfemail="HEX">...</a>
CF_EMAIL_RE = re.compile(r"^[0-9a-fA-F]{4,}$")


def cf_decode_email(cfemail: str) -> str:
    raw = bytes.fromhex(cfemail)
    key = raw[0]
    return "".join(chr(b ^ key) for b in raw[1:])


def decode_cf_emails(soup: BeautifulSoup) -> None:
    """Replace Cloudflare's obfuscated-email placeholders with real addresses."""
    for el in soup.find_all(attrs={"data-cfemail": True}):
        cfemail = el.get("data-cfemail", "")
        if CF_EMAIL_RE.match(cfemail):
            try:
                el.string = cf_decode_email(cfemail)
            except ValueError:
                pass
    for el in soup.select('a[href*="cdn-cgi/l/email-protection"]'):
        href = el.get("href", "")
        if "#" in href:
            cfemail = href.split("#", 1)[1]
            if CF_EMAIL_RE.match(cfemail):
                try:
                    el.string = cf_decode_email(cfemail)
                except ValueError:
                    pass


LOWERCASE_CONNECTORS = {"of", "and", "the", "for"}


def logo_filename_to_name(src: str) -> str:
    """'World-of-Hyatt.svg' -> 'World of Hyatt', 'BASF.svg' -> 'BASF'."""
    stem = Path(urllib.parse.urlparse(src).path).stem
    stem = re.sub(r"[-_]+", " ", stem).strip()
    # Preserve short all-caps/mixed-case acronyms (BASF, TATA) as-is; title-case
    # ordinary words, keeping connectors lowercase unless they open the name.
    words = stem.split()
    out = []
    for i, w in enumerate(words):
        if w.isupper() and len(w) <= 5:
            out.append(w)
        elif w.islower() and len(w) <= 3 and len(words) == 1:
            # Lone short lowercase filenames are acronyms in practice (abp.svg).
            out.append(w.upper())
        elif w.lower() in LOWERCASE_CONNECTORS and i > 0:
            out.append(w.lower())
        else:
            out.append(w.capitalize())
    return " ".join(out)


def extract_client_logos(soup: BeautifulSoup) -> list:
    """Client/customer logo carousels (e.g. class='owl-carousel client-slid')
    only ever carry the company name in the image filename -- alt text is
    generic ('AI access control' on every logo) and there's no visible text
    at all. markdownify's img stripping would otherwise silently drop this
    entirely, so pull it out here before that happens."""
    names = []
    seen = set()
    for container in soup.select('[class*="client"]'):
        for img in container.find_all("img"):
            src = img.get("src", "")
            if not src or not re.search(r"\.(svg|png|jpe?g|webp)(\?|$)", src, re.I):
                continue
            name = logo_filename_to_name(src)
            if name and name.lower() not in seen and name.lower() != "idcube":
                seen.add(name.lower())
                names.append(name)
    return names


def html_to_clean_markdown(html: str, page_url: str = BASE, pdf_sink=None) -> str:
    """Convert a page to Markdown, stripping only genuinely non-visible tags.
    Deliberately does NOT try to guess a 'main content' region -- this site's
    broken tag semantics make that unreliable. Boilerplate (nav/footer/modals)
    is removed afterwards, across the whole crawl, in strip_boilerplate_lines()."""
    soup = BeautifulSoup(html, "lxml")
    decode_cf_emails(soup)
    inline_pdf_links(soup, page_url, pdf_sink)
    for tag in list(soup.find_all(NON_VISIBLE_TAGS)):
        if tag.parent is not None:
            tag.decompose()
    body = soup.body or soup
    # escape_underscores=False: the default escaping turns URLs like
    # GreenID_brochure.pdf into GreenID\_brochure.pdf -- a broken link.
    markdown = html_to_markdown(
        str(body), heading_style="ATX", strip=["a", "img"], escape_underscores=False
    )
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return markdown


def extract_metadata(html: str, url: str):
    """Title/description via trafilatura (handles og:title/meta fallbacks,
    date extraction) -- this part of trafilatura works fine independent of
    its body-extraction issues on this site."""
    try:
        meta = trafilatura.extract_metadata(html, default_url=url)
    except Exception:
        meta = None
    title = getattr(meta, "title", None)
    description = getattr(meta, "description", None)
    date = getattr(meta, "date", None)
    if not title:
        soup = BeautifulSoup(html, "lxml")
        if soup.title:
            title = soup.title.get_text(strip=True)
    return title, description, date


def markdown_to_text(markdown: str) -> str:
    text = re.sub(r"!\[[^\]]*\]", "", markdown)  # leftover image alt markers, if any
    text = re.sub(r"[#*_`>]", "", text)
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.MULTILINE)
    text = "\n".join(ln.strip() for ln in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


REGION_PREFIXES = ("in", "us", "mea")


def region_of(url: str) -> str:
    """'/in/en/products/...' -> 'in'; unprefixed URLs are the global site."""
    first = urllib.parse.urlparse(url).path.strip("/").split("/")[0]
    return first if first in REGION_PREFIXES else "global"


def build_boilerplate_lines(raw_pages: list, min_pages: int = 5, fraction: float = 0.4) -> set:
    """A line that shows up verbatim on a large fraction of *distinct* pages
    is site chrome (nav menu, region selector, footer, modal placeholders),
    not real content -- real content is page-specific by definition. Skipped
    entirely on tiny crawls (< min_pages) where frequency isn't meaningful."""
    total = len(raw_pages)
    if total < min_pages:
        return set()

    line_pages = defaultdict(set)
    for rec in raw_pages:
        seen_this_page = set()
        for line in rec["raw_markdown"].splitlines():
            line = line.strip()
            if not line or line in seen_this_page:
                continue
            seen_this_page.add(line)
            line_pages[line].add(rec["url"])

    threshold = max(min_pages, int(fraction * total))
    return {line for line, urls in line_pages.items() if len(urls) >= threshold}


def strip_boilerplate(markdown: str, boilerplate: set) -> str:
    kept = [ln for ln in markdown.splitlines() if ln.strip() not in boilerplate]
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# Unfilled demo/placeholder content some template-driven sites leave behind
# verbatim in production (this site included: a Sales-team widget still
# carrying its original Elementor sample data on some pages). Not
# boilerplate in the nav/footer sense -- just junk baked into that one page's
# HTML -- so it wouldn't be caught by the cross-page frequency pass above.
PLACEHOLDER_LINE_RE = re.compile(
    r"info@example\.com|lorem ipsum|duden flows|\+1 \(859\) 254-6589",
    re.IGNORECASE,
)


def strip_placeholder_blocks(markdown: str) -> str:
    """Drop any small block (line +/-3 neighbors) containing a known
    placeholder marker, since these show up as a few-line card, not a
    single isolated line."""
    lines = markdown.splitlines()
    junk_idx = {i for i, ln in enumerate(lines) if PLACEHOLDER_LINE_RE.search(ln)}
    if not junk_idx:
        return markdown
    drop = set()
    for i in junk_idx:
        drop.update(range(max(0, i - 3), min(len(lines), i + 4)))
    kept = [ln for i, ln in enumerate(lines) if i not in drop]
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def crawl_pages(urls_with_lastmod, session, out_dir: Path, manifest: dict, delay: float, log_file, pdf_sink=None):
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: fetch + convert every page to raw (un-deduplicated) markdown.
    raw_pages = []
    failed = 0
    for i, (url, lastmod) in enumerate(urls_with_lastmod, 1):
        log(f"[fetch {i}/{len(urls_with_lastmod)}] {url}", log_file)
        resp = fetch(url, session)
        if resp is None or resp.status_code != 200:
            log(f"  ! skip (status={getattr(resp, 'status_code', None)})", log_file)
            failed += 1
            continue
        raw_markdown = html_to_clean_markdown(resp.text, url, pdf_sink)
        if len(raw_markdown) < 60:
            log("  ! skip (no content)", log_file)
            failed += 1
            continue
        title, description, date = extract_metadata(resp.text, url)
        # Extracted separately from raw_markdown (not appended to it) because
        # this carousel is sitewide -- if it went through the normal
        # boilerplate-frequency pass below, it would get stripped right back
        # out as boilerplate along with the nav/footer.
        client_names = extract_client_logos(BeautifulSoup(resp.text, "lxml"))
        raw_pages.append({
            "url": url, "lastmod": lastmod, "raw_markdown": raw_markdown,
            "title": title, "description": description, "date": date,
            "client_names": client_names,
        })
        time.sleep(delay)

    # Pass 2: figure out what's boilerplate, then write each page with it
    # stripped. Done per region as well as sitewide: each region (in/us/mea/
    # global) has its own nav menu and footer, which appears on only ~1/4 of a
    # full-site crawl -- below the sitewide frequency threshold -- so a single
    # sitewide pass would leave every region's menu in its pages' content.
    sitewide = build_boilerplate_lines(raw_pages)
    by_region = defaultdict(list)
    for rec in raw_pages:
        by_region[region_of(rec["url"])].append(rec)
    region_boilerplate = {
        region: sitewide | build_boilerplate_lines(recs) for region, recs in by_region.items()
    }
    for region, lines in sorted(region_boilerplate.items()):
        log(f"Boilerplate for '{region}': {len(lines)} lines across {len(by_region[region])} pages", log_file)

    new_manifest = {}
    ok, skipped = 0, 0
    for rec in raw_pages:
        markdown = strip_boilerplate(rec["raw_markdown"], region_boilerplate[region_of(rec["url"])])
        markdown = strip_placeholder_blocks(markdown)
        # Attach only to the homepage: the client-logo carousel is sitewide,
        # so every page's raw_pages entry carries the same client_names list
        # -- repeating it verbatim on all of them would just be duplicate
        # chunks in the index for no benefit.
        # Written as a sentence, not a bare list: a list of brand names embeds
        # nowhere near queries like "who are your customers?", so it never
        # ranked in retrieval.
        if rec["url"] == HOMEPAGE_URL and rec.get("client_names"):
            names = rec["client_names"]
            joined = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
            markdown += (
                "\n\n## IDCUBE customers and clients\n\n"
                "Leading companies and organisations that are IDCUBE customers and use "
                f"IDCUBE access control solutions include {joined}."
            )
        if len(markdown.strip()) < 40:
            log(f"  ! skip {rec['url']} (empty after boilerplate removal)", log_file)
            skipped += 1
            continue
        text = markdown_to_text(markdown)
        h = content_hash(markdown)
        prev = manifest.get(rec["url"])
        changed = prev is None or prev.get("hash") != h

        record = {
            "url": rec["url"],
            "title": rec["title"],
            "description": rec["description"],
            "date": rec["date"],
            "lastmod_sitemap": rec["lastmod"],
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": h,
            "changed_since_last_run": changed,
            "markdown": markdown,
            "text": text,
            "word_count": len(text.split()),
        }
        slug = slugify(rec["url"])
        write_json(pages_dir / f"{slug}.json", record)
        new_manifest[rec["url"]] = {
            "hash": h, "lastmod": rec["lastmod"],
            "scraped_at": record["scraped_at"], "slug": slug,
        }
        ok += 1

    log(f"Pages done: {ok} ok, {skipped} empty-after-cleanup, {failed} failed", log_file)
    return new_manifest


def crawl_faq_page(session, out_dir: Path, manifest: dict, log_file):
    """Special-cased: the FAQ page's Q&A content only exists as an inline
    JSON blob (see FAQ_CONFIG_RE), not in the rendered HTML text, so it can't
    go through crawl_pages()'s normal markdown pipeline. One markdown
    heading per question keeps each Q&A as its own chunk after chunker.py."""
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    log(f"[faq] fetching {FAQ_URL}", log_file)
    resp = fetch(FAQ_URL, session)
    if resp is None or resp.status_code != 200:
        log(f"  ! faq fetch failed (status={getattr(resp, 'status_code', None)})", log_file)
        return {}

    match = FAQ_CONFIG_RE.search(resp.text)
    if not match:
        log("  ! faqConfig JSON not found in page -- site markup may have changed", log_file)
        return {}

    try:
        config = json.loads(match.group(1))
        faqs = config["faqs"]
    except (json.JSONDecodeError, KeyError) as e:
        log(f"  ! failed to parse faqConfig JSON: {e}", log_file)
        return {}

    if not faqs:
        log("  ! faqConfig had zero FAQs -- skipping", log_file)
        return {}

    by_category = defaultdict(list)
    for item in faqs:
        cat = (item.get("cat_names") or ["General"])[0]
        by_category[cat].append(item)

    lines = ["# Physical Access Control System (PACS) FAQs"]
    for cat in sorted(by_category):
        lines.append(f"\n## {cat}")
        for item in by_category[cat]:
            question = item["question"].strip()
            answer = item["answer"].strip()
            lines.append(f"\n### {question}\n\n{answer}")
    markdown = "\n".join(lines)
    text = markdown_to_text(markdown)
    h = content_hash(markdown)

    prev = manifest.get(FAQ_URL)
    changed = prev is None or prev.get("hash") != h
    record = {
        "url": FAQ_URL,
        "title": "Physical Access Control System (PACS) FAQs | IDCUBE",
        "description": f"{len(faqs)} frequently asked questions about IDCUBE products and solutions.",
        "date": None,
        "lastmod_sitemap": None,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "content_hash": h,
        "changed_since_last_run": changed,
        "markdown": markdown,
        "text": text,
        "word_count": len(text.split()),
    }
    slug = slugify(FAQ_URL)
    write_json(pages_dir / f"{slug}.json", record)
    log(f"[faq] wrote {len(faqs)} FAQs across {len(by_category)} categories", log_file)
    return {FAQ_URL: {"hash": h, "lastmod": None, "scraped_at": record["scraped_at"], "slug": slug}}


def crawl_pdfs(session, out_dir: Path, delay: float, log_file, limit=None, extra_urls=(), listed_sink=None):
    try:
        from pypdf import PdfReader
    except ImportError:
        log("  ! pypdf not installed, skipping PDFs (pip install pypdf)", log_file)
        return {}

    seen = set()
    pdf_urls = resolve_sitemap(BASE + PDF_SITEMAP, session, seen, log_file)
    listed = {u for u, _ in pdf_urls}
    unlisted = sorted(set(extra_urls) - listed)
    if unlisted:
        log(f"  {len(unlisted)} PDFs linked from pages but missing from the PDF sitemap", log_file)
    pdf_urls += [(u, None) for u in unlisted]
    if listed_sink is not None:
        listed_sink.update(u for u, _ in pdf_urls)
    if limit:
        pdf_urls = pdf_urls[:limit]

    pdfs_dir = out_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for i, (url, lastmod) in enumerate(pdf_urls, 1):
        log(f"[pdf {i}/{len(pdf_urls)}] {url}", log_file)
        resp = fetch(url, session)
        if resp is None or resp.status_code != 200:
            log(f"  ! skip (status={getattr(resp, 'status_code', None)})", log_file)
            continue
        try:
            reader = PdfReader(io.BytesIO(resp.content))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages).strip()
        except Exception as e:
            log(f"  ! failed to parse PDF: {e}", log_file)
            continue
        if len(text) < 40:
            continue
        h = content_hash(text)
        slug = slugify(url)
        record = {
            "url": url,
            "lastmod_sitemap": lastmod,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": h,
            "text": text,
            "word_count": len(text.split()),
        }
        write_json(pdfs_dir / f"{slug}.json", record)
        results[url] = {"hash": h, "lastmod": lastmod, "slug": slug}
        time.sleep(delay)

    return results


MAX_PRUNE_FRACTION = 0.2


def prune_removed(folder: Path, keep_urls: set, log_file) -> int:
    """Delete saved documents whose URL the site no longer lists, so pages
    removed from idcubesystems.com stop being indexed and cited. Keyed on
    what the sitemaps/pages list this run -- not on fetch success -- so a
    page that merely failed to load keeps last week's copy. Refuses to
    delete more than MAX_PRUNE_FRACTION in one run: a mass removal means the
    sitemap itself failed (e.g. a Cloudflare block), not that the site
    deleted most of its pages."""
    files = [f for f in folder.glob("*.json") if not f.name.startswith("documents-catalog-")]
    stale = []
    for f in files:
        try:
            url = json.loads(f.read_text(encoding="utf-8")).get("url")
        except (ValueError, OSError):
            continue
        if url and url not in keep_urls:
            stale.append((f, url))
    if files and len(stale) > MAX_PRUNE_FRACTION * len(files):
        log(f"  ! NOT pruning {folder.name}/: {len(stale)} of {len(files)} would be removed -- "
            f"looks like a failed crawl, check the sitemap fetch", log_file)
        return 0
    for f, url in stale:
        f.unlink()
        log(f"  - removed (no longer on site): {url}", log_file)
    return len(stale)


def upload_month(url: str) -> str:
    """'/wp-content/uploads/2025/09/x.pdf' -> '2025/09' (sorts newest last)."""
    m = re.search(r"/uploads/(\d{4}/\d{2})/", url)
    return m.group(1) if m else ""


def write_documents_catalog(pdf_urls, out_dir: Path, log_file) -> int:
    """One page per document category listing every PDF with its exact
    download link -- one heading per document, so each becomes its own
    chunk and "send me the GreenID brochure" retrieves that exact line.
    Only PDFs that downloaded successfully this run are listed, so the
    catalog never carries a dead link."""
    pages_dir = out_dir / "pages"
    for old in pages_dir.glob("documents-catalog-*.json"):
        old.unlink()  # rebuilt below; drops categories that no longer have documents
    newest = {}
    for url in pdf_urls:
        key = document_label(url)
        current = newest.get(key)
        # Same document uploaded more than once: keep the latest upload.
        if current is None or (upload_month(url), -len(url)) > (upload_month(current), -len(current)):
            newest[key] = url

    by_category = defaultdict(list)
    for (category, label), url in newest.items():
        by_category[category].append((label, url))

    written = 0
    for category, docs in by_category.items():
        cat_title, hub_url = DOC_CATEGORIES[category]
        lines = [f"# IDCUBE downloadable documents: {cat_title}"]
        for label, url in sorted(docs, key=lambda d: d[0].lower()):
            lines.append(
                f"\n## {label}\n\nDirect PDF download link for the IDCUBE {label} "
                f"({cat_title.lower()}): {url}"
            )
        markdown = "\n".join(lines)
        text = markdown_to_text(markdown)
        write_json(pages_dir / f"documents-catalog-{category}.json", {
            "url": hub_url,
            "title": f"IDCUBE {cat_title} (PDF downloads)",
            "description": f"Direct download links for {len(docs)} IDCUBE {cat_title.lower()}.",
            "date": None,
            "lastmod_sitemap": None,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": content_hash(markdown),
            "changed_since_last_run": True,
            "markdown": markdown,
            "text": text,
            "word_count": len(text.split()),
        })
        written += len(docs)
    log(f"[documents] catalog lists {written} PDFs across {len(by_category)} categories", log_file)
    return written


def main():
    ap = argparse.ArgumentParser(description="Crawl idcubesystems.com for RAG ingestion")
    ap.add_argument("--region", choices=["global", "in", "us", "mea", "all"], default="in",
                     help="Which sitemap set to crawl (default: in = India site)")
    ap.add_argument("--out", default="data", help="Output directory")
    ap.add_argument("--delay", type=float, default=1.5, help="Seconds between page requests")
    ap.add_argument("--include-pdf", action="store_true", help="Also crawl PDFs from sitemap-pdf.xml")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of pages (for a quick test run)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "crawl_log.txt"

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    session = requests.Session()

    regions = ["global", "in", "us", "mea"] if args.region == "all" else [args.region]
    seen_sitemaps = set()
    all_urls = {}
    for region in regions:
        for sm in SITEMAP_SETS[region]:
            log(f"Resolving sitemap set '{region}': {BASE + sm}", log_file)
            for url, lastmod in resolve_sitemap(BASE + sm, session, seen_sitemaps, log_file):
                all_urls[url] = lastmod

    for url in EXTRA_URLS:
        all_urls.setdefault(url, None)

    # The global sitemap index includes sitemap-pdf.xml, so PDF URLs land in
    # this list too -- crawl_pdfs() handles those; the HTML pipeline would
    # just produce garbage from PDF bytes. FAQ_URL has its own handler.
    urls_with_lastmod = [
        (u, lm) for u, lm in all_urls.items()
        if not urllib.parse.urlparse(u).path.lower().endswith(".pdf") and u != FAQ_URL
    ]
    if args.limit:
        urls_with_lastmod = urls_with_lastmod[: args.limit]
    log(f"Total unique pages to crawl: {len(urls_with_lastmod)}", log_file)

    linked_pdfs = set()
    page_manifest = crawl_pages(
        urls_with_lastmod, session, out_dir, manifest, args.delay, log_file, pdf_sink=linked_pdfs
    )
    manifest.update(page_manifest)

    faq_manifest = crawl_faq_page(session, out_dir, manifest, log_file)
    manifest.update(faq_manifest)

    listed_pdfs = set()
    if args.include_pdf:
        pdf_manifest = crawl_pdfs(
            session, out_dir, args.delay, log_file, limit=args.limit,
            extra_urls=linked_pdfs, listed_sink=listed_pdfs,
        )
        manifest.update(pdf_manifest)
        write_documents_catalog(pdf_manifest.keys(), out_dir, log_file)

    if not args.limit:  # a --limit test run sees only part of the site
        removed = prune_removed(out_dir / "pages", {u for u, _ in urls_with_lastmod} | {FAQ_URL}, log_file)
        if args.include_pdf:
            removed += prune_removed(out_dir / "pdfs", listed_pdfs, log_file)
        log(f"Pruned {removed} documents no longer on the site", log_file)

    write_json(manifest_path, manifest)

    changed = sum(1 for v in page_manifest.values() if v)
    log(f"Done. Manifest has {len(manifest)} entries. See {out_dir}/pages/*.json", log_file)


if __name__ == "__main__":
    sys.exit(main())
