"""
Server-side relay between the website chat widget (widget.js) and the
Purple Fabric "IDCUBE Website Assistant" agent (see openApi.json).

Why a relay: Purple Fabric authenticates with an apikey + username +
password. Anything shipped to the browser is public, so those credentials
must stay on this server. The browser only ever gets an opaque, signed
session token and the agent's reply text -- never the credentials, the
bearer token, Purple Fabric's internal IDs, traces, or cost metrics.

Configuration (environment variables):
  PF_APIKEY, PF_USERNAME, PF_PASSWORD   required -- Purple Fabric credentials
  PF_BASE_URL        default https://api.in.intellectseecstag.com
  PF_ASSET_ID        default: the IDCUBE Website Assistant asset version
  ALLOWED_ORIGINS    comma-separated site origins allowed to call /chat/*
  SESSION_SECRET     signs session tokens; random per process if unset
                     (tokens then just expire on restart and the widget
                     transparently starts a new session)

Endpoints (all under /chat):
  POST /chat/session              -> {"session": "<signed token>"}
  GET  /chat/starters             -> {"starters": ["...", ...]}
  POST /chat/message  {"session", "query"}  -> text/event-stream of
       data: {"type": "delta", "text": "..."}   streamed fragments
       data: {"type": "final", "text": "..."}   consolidated markdown reply
       data: {"type": "error", "text": "..."}   visitor-safe error message
       data: {"type": "end"}                    reply complete
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from collections import defaultdict, deque

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

log = logging.getLogger("chat_proxy")

PF_BASE_URL = os.environ.get("PF_BASE_URL", "https://api.in.intellectseecstag.com").rstrip("/")
PF_ASSET_ID = os.environ.get("PF_ASSET_ID", "060ffec6-233c-4cc0-a345-2a603e9dfb07")
PF_TOKEN_PATH = "/accesstoken/pfindidcube"

ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS", "https://www.idcubesystems.com,https://idcubesystems.com"
    ).split(",") if o.strip()
]

SESSION_SECRET = (os.environ.get("SESSION_SECRET") or secrets.token_hex(32)).encode()
OBJECT_ID_RE = re.compile(r"^[a-fA-F0-9]{24}$")

MAX_QUERY_CHARS = 1000
# Every message costs LLM spend, and this endpoint is public -- cap per visitor.
RATE_LIMIT_MESSAGES = 20
RATE_LIMIT_WINDOW_S = 600
FALLBACK_ERROR = (
    "Sorry, I'm having trouble answering right now. Please try again in a moment, "
    "or reach our team at contact@idcubesystems.com."
)

router = APIRouter(prefix="/chat")


def _credentials():
    missing = [k for k in ("PF_APIKEY", "PF_USERNAME", "PF_PASSWORD") if not os.environ.get(k)]
    if missing:
        log.error("Chat relay not configured; missing env vars: %s", ", ".join(missing))
        raise HTTPException(status_code=503, detail="Chat is not configured")
    return os.environ["PF_APIKEY"], os.environ["PF_USERNAME"], os.environ["PF_PASSWORD"]


class _TokenCache:
    """Purple Fabric bearer tokens last `expires_in` seconds (3600 today);
    reuse one until shortly before expiry instead of logging in per message."""

    def __init__(self):
        self._token = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    async def get(self, client: httpx.AsyncClient, force_refresh: bool = False) -> str:
        async with self._lock:
            if not force_refresh and self._token and time.monotonic() < self._expires_at:
                return self._token
            apikey, username, password = _credentials()
            resp = await client.get(
                PF_BASE_URL + PF_TOKEN_PATH,
                headers={"apikey": apikey, "username": username, "password": password},
            )
            if resp.status_code != 200:
                log.error("Purple Fabric login failed: HTTP %s %s", resp.status_code, resp.text[:300])
                raise HTTPException(status_code=502, detail="Upstream login failed")
            body = resp.json()
            self._token = body["access_token"]
            ttl = int(body.get("expires_in") or 3600)
            self._expires_at = time.monotonic() + max(ttl - 120, 60)
            return self._token


_tokens = _TokenCache()
_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0))


async def _pf_request(method: str, path: str, **kwargs) -> httpx.Response:
    """Authenticated call; on 401 the token is refreshed once and retried."""
    apikey = _credentials()[0]
    extra_headers = kwargs.pop("headers", {})
    for attempt in (1, 2):
        token = await _tokens.get(_client, force_refresh=(attempt == 2))
        headers = {"Authorization": f"Bearer {token}", "apikey": apikey, **extra_headers}
        resp = await _client.request(method, PF_BASE_URL + path, headers=headers, **kwargs)
        if resp.status_code != 401:
            return resp
    return resp


# --- visitor session tokens -------------------------------------------------
# The browser holds "<pf_session_id>.<hmac>" rather than a bare Purple
# Fabric session id, so visitors can't post into sessions they didn't open.

def _sign(session_id: str) -> str:
    mac = hmac.new(SESSION_SECRET, session_id.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{session_id}.{mac}"


def _verify(token: str) -> str:
    session_id, _, mac = (token or "").partition(".")
    if not OBJECT_ID_RE.match(session_id):
        raise HTTPException(status_code=401, detail="Invalid session")
    expected = hmac.new(SESSION_SECRET, session_id.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(mac, expected):
        raise HTTPException(status_code=401, detail="Invalid session")
    return session_id


# --- per-visitor rate limit ---------------------------------------------------

_recent = defaultdict(deque)


def _client_ip(request: Request) -> str:
    # Render terminates TLS and forwards the visitor's address.
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _check_rate_limit(request: Request):
    now = time.monotonic()
    if len(_recent) > 5000:
        for ip in [ip for ip, w in _recent.items() if not w or now - w[-1] > RATE_LIMIT_WINDOW_S]:
            del _recent[ip]
    window = _recent[_client_ip(request)]
    while window and now - window[0] > RATE_LIMIT_WINDOW_S:
        window.popleft()
    if len(window) >= RATE_LIMIT_MESSAGES:
        raise HTTPException(status_code=429, detail="Too many messages")
    window.append(now)


# --- endpoints ------------------------------------------------------------------

@router.post("/session")
async def create_session():
    resp = await _pf_request(
        "POST",
        f"/purplefabric/v1/interaction/{PF_ASSET_ID}/sessions",
        json={"session_name": f"website-{secrets.token_hex(4)}"},
    )
    if resp.status_code not in (200, 201):
        log.error("Create session failed: HTTP %s %s", resp.status_code, resp.text[:300])
        raise HTTPException(status_code=502, detail="Could not start a chat")
    return {"session": _sign(resp.json()["session_id"])}


_starters_cache = {"at": 0.0, "value": []}


@router.get("/starters")
async def starters():
    if time.monotonic() - _starters_cache["at"] < 600:
        return {"starters": _starters_cache["value"]}
    try:
        resp = await _pf_request("GET", f"/purplefabric/v1/interaction/{PF_ASSET_ID}/session-starters")
        value = resp.json().get("session_starters", []) if resp.status_code == 200 else []
    except (httpx.HTTPError, ValueError) as e:
        log.warning("Starters unavailable: %s", e)
        value = []
    _starters_cache.update(at=time.monotonic(), value=value)
    return {"starters": value}


class MessageRequest(BaseModel):
    session: str
    query: str


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/message")
async def message(req: MessageRequest, request: Request):
    session_id = _verify(req.session)
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty message")
    if len(query) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=400, detail=f"Message too long (max {MAX_QUERY_CHARS} characters)")
    _check_rate_limit(request)
    apikey = _credentials()[0]
    url = f"{PF_BASE_URL}/purplefabric/v1/interaction/sessions/{session_id}/messages"

    async def relay():
        try:
            for attempt in (1, 2):
                token = await _tokens.get(_client, force_refresh=(attempt == 2))
                async with _client.stream(
                    "POST",
                    url,
                    headers={"Authorization": f"Bearer {token}", "apikey": apikey, "Accept": "text/event-stream"},
                    json={"query": query, "response_mode": "stream"},
                ) as resp:
                    if resp.status_code == 401 and attempt == 1:
                        continue  # token expired server-side; log in again and retry once
                    if resp.status_code != 200:
                        body = (await resp.aread())[:300]
                        log.error("Send message failed: HTTP %s %s", resp.status_code, body)
                        yield _sse({"type": "error", "text": FALLBACK_ERROR})
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        try:
                            chunk = json.loads(line[5:].strip())
                        except ValueError:
                            continue
                        event = chunk.get("event")
                        content = chunk.get("content")
                        if event == "LLM_RESPONSE_STREAM" and isinstance(content, str) and content:
                            yield _sse({"type": "delta", "text": content})
                        elif event == "FINAL_RESPONSE":
                            text = content.get("response") if isinstance(content, dict) else None
                            if text:
                                yield _sse({"type": "final", "text": text})
                        elif event == "ERROR":
                            log.error("Agent error chunk: %s", json.dumps(chunk)[:500])
                            yield _sse({"type": "error", "text": FALLBACK_ERROR})
                            return
                        elif event == "STREAM_END":
                            break
                        else:
                            # init / MESSAGE_DETAILS / heartbeat: keep the browser
                            # connection alive through proxies without sending content.
                            yield ": keep-alive\n\n"
                    yield _sse({"type": "end"})
                    return
        except httpx.HTTPError as e:
            log.error("Relay stream failed: %s", e)
            yield _sse({"type": "error", "text": FALLBACK_ERROR})

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
