"""
Server-side relay between the website chat widget (widget.js) and AIsha's
LangChain agent (aisha_agent.py), which answers from IDCUBE's own content.

Why a relay: the model's API key must stay on this server. The browser only
ever gets an opaque, signed session token and the reply text -- never the
key, the instructions, the search results or token usage.

Configuration (environment variables):
  LLM_MODEL          LangChain "provider:model", default anthropic:claude-haiku-5-5
  ANTHROPIC_API_KEY  required for the default model
  DAILY_MESSAGE_CAP  visitor messages answered per day (IST), all visitors
                     together; default 300
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

import datetime
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.exceptions import ModelRefusalError
from pydantic import BaseModel

import aisha_agent

log = logging.getLogger("chat_proxy")

DAILY_MESSAGE_CAP = int(os.environ.get("DAILY_MESSAGE_CAP", "300"))

ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS", "https://www.idcubesystems.com,https://idcubesystems.com"
    ).split(",") if o.strip()
]

SESSION_SECRET = (os.environ.get("SESSION_SECRET") or secrets.token_hex(32)).encode()
SESSION_ID_RE = re.compile(r"^[a-f0-9]{24}$")

MAX_QUERY_CHARS = 1000
# Every message costs LLM spend, and this endpoint is public -- cap per visitor.
RATE_LIMIT_MESSAGES = 20
RATE_LIMIT_WINDOW_S = 600
FALLBACK_ERROR = (
    "Sorry, I'm having trouble answering right now. Please try again in a moment, "
    "or reach our team at contact@idcubesystems.com."
)
DAILY_CAP_REPLY = (
    "Sorry, I've answered a lot of questions today and need a short break. "
    "Please try again tomorrow, or reach our team at contact@idcubesystems.com."
)
REFUSAL_REPLY = (
    "Sorry, I can't help with that one. I'd be glad to help with anything about "
    "IDCUBE's products, a demo or support, though."
)
STARTERS = [
    "What is GreenID?",
    "Tell me about Access360.AI",
    "Do you have a visitor management system?",
    "How can I book a demo?",
]

_agent = None


def _get_agent(request: Request):
    global _agent
    if _agent is None:
        try:
            aisha_agent.check_configured()
            _agent = aisha_agent.build_agent(request.app.state.collection)
        except Exception as e:  # missing key, unknown provider or model name
            log.error("Chat not configured (%s): %s", aisha_agent.LLM_MODEL, e)
            raise HTTPException(status_code=503, detail="Chat is not configured")
    return _agent


router = APIRouter(prefix="/chat")


# --- visitor session tokens -------------------------------------------------
# The browser holds "<session_id>.<hmac>" rather than a bare id, so visitors
# can't post into sessions they didn't open.

def _sign(session_id: str) -> str:
    mac = hmac.new(SESSION_SECRET, session_id.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{session_id}.{mac}"


def _verify(token: str) -> str:
    session_id, _, mac = (token or "").partition(".")
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=401, detail="Invalid session")
    expected = hmac.new(SESSION_SECRET, session_id.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(mac, expected):
        raise HTTPException(status_code=401, detail="Invalid session")
    return session_id


# --- conversation memory ----------------------------------------------------------
# Text-only turns per session, in memory: enough for follow-up questions
# ("does it support QR codes?"). Search results aren't kept, which keeps every
# later message cheap. Lost on restart, which only costs context.

MAX_HISTORY_MESSAGES = 8
SESSION_IDLE_S = 3600
MAX_SESSIONS = 5000
_histories = {}


def _history(session_id: str) -> list:
    entry = _histories.get(session_id)
    if not entry or time.monotonic() - entry["at"] > SESSION_IDLE_S:
        return []
    return list(entry["messages"])


def _remember(session_id: str, question: str, answer: str):
    now = time.monotonic()
    if len(_histories) > MAX_SESSIONS:
        for sid in [s for s, e in _histories.items() if now - e["at"] > SESSION_IDLE_S]:
            del _histories[sid]
        while len(_histories) > MAX_SESSIONS:
            del _histories[next(iter(_histories))]
    messages = _history(session_id) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    _histories[session_id] = {"at": now, "messages": messages[-MAX_HISTORY_MESSAGES:]}


# --- per-visitor rate limit and daily cap -------------------------------------

_recent = defaultdict(deque)
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
_daily = {"day": None, "count": 0}


def _client_ip(request: Request) -> str:
    # The host's proxy terminates TLS and forwards the visitor's address.
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


def _take_daily_slot() -> bool:
    """A hard ceiling on spend: one shared count of answered messages per IST day."""
    today = datetime.datetime.now(IST).date()
    if _daily["day"] != today:
        _daily.update(day=today, count=0)
    if _daily["count"] >= DAILY_MESSAGE_CAP:
        return False
    _daily["count"] += 1
    return True


# --- endpoints ------------------------------------------------------------------

@router.post("/session")
async def create_session():
    return {"session": _sign(secrets.token_hex(12))}


@router.get("/starters")
async def starters():
    return {"starters": STARTERS}


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
    agent = _get_agent(request)
    messages = _history(session_id) + [{"role": "user", "content": query}]

    async def relay():
        if not _take_daily_slot():
            log.warning("Daily cap of %d messages reached", DAILY_MESSAGE_CAP)
            yield _sse({"type": "error", "text": DAILY_CAP_REPLY})
            return
        started, usage, answer = time.monotonic(), {}, ""
        try:
            async for kind, text in aisha_agent.stream_reply(agent, messages, usage):
                if kind == "delta":
                    yield _sse({"type": "delta", "text": text})
                elif kind == "keepalive":
                    # Keeps the browser connection open through proxies.
                    yield ": keep-alive\n\n"
                else:
                    answer = text
        except ModelRefusalError:
            log.warning("Model refused (session %s)", session_id[:8])
            answer = REFUSAL_REPLY
        except Exception:  # provider outage, timeout, bad key: never shown to visitors
            log.exception("Agent run failed")
            yield _sse({"type": "error", "text": FALLBACK_ERROR})
            return
        log.info("chat %s: %.1fs %s", session_id[:8], time.monotonic() - started, json.dumps(usage))
        if not answer:
            yield _sse({"type": "error", "text": FALLBACK_ERROR})
            return
        if answer != REFUSAL_REPLY:
            _remember(session_id, query, answer)
        yield _sse({"type": "final", "text": answer})
        yield _sse({"type": "end"})

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
