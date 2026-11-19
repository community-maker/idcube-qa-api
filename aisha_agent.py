"""
AIsha's agent, built with LangChain.

One chat model with one tool, search_website, which answers from the local
index (ask.retrieve). LangChain runs the loop: the model searches, reads the
excerpts, and writes the reply.

The model is a setting. LLM_MODEL takes LangChain's "provider:model" form:
  anthropic:claude-haiku-5-5   default; needs ANTHROPIC_API_KEY
For Claude on Microsoft Foundry, also set ANTHROPIC_BASE_URL to
https://<resource>.services.ai.azure.com/anthropic and use the Foundry key.
Another provider needs its LangChain package (e.g. langchain-openai) in
requirements.txt and its own API key.
"""

import asyncio
import os
import re
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelRefusalMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from ask import build_context, retrieve

LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic:claude-haiku-5-5")


def _system_prompt():
    """AIsha's rules, plus the team's example conversations (aisha_examples.txt)
    if any. "#" lines in the examples file are notes for whoever edits it."""
    prompt = Path(__file__).with_name("aisha_prompt.txt").read_text(encoding="utf-8")
    examples_file = Path(__file__).with_name("aisha_examples.txt")
    if examples_file.exists():
        examples = "\n".join(line for line in examples_file.read_text(encoding="utf-8").splitlines()
                             if not line.startswith("#") or line.startswith("###")).strip()
        if examples:
            prompt += (
                "\n\n17. EXAMPLE CONVERSATIONS\n"
                "- These show the tone, length and format to aim for. They are not part of the "
                "conversation; never mention them. Facts can change: still search before every "
                "answer and use this turn's search results.\n\n" + examples + "\n"
            )
    return prompt


SYSTEM_PROMPT = _system_prompt()
MAX_SEARCH_QUERY_CHARS = 300
# Up to one search per part of a question (comparisons, several product
# types), then the answer.
MAX_SEARCHES = 3
MAX_MODEL_CALLS = 5


def check_configured():
    """Raise ValueError if the model's API key is missing, before any visitor waits on it."""
    if LLM_MODEL.startswith("anthropic:") and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY is not set")


def _chat_model():
    kwargs = {"max_tokens": 1500, "timeout": 60, "max_retries": 2}
    if LLM_MODEL.startswith("anthropic:"):
        # Short, grounded answers: no thinking, low effort -- fastest and cheapest.
        kwargs.update(thinking={"type": "disabled"}, output_config={"effort": "low"})
    return init_chat_model(LLM_MODEL, **kwargs)


# Messages that need no facts, so no search: greetings, thanks, acknowledgements.
SMALL_TALK_RE = re.compile(
    r"(hi+|hello|hey|namaste|good (morning|afternoon|evening)|thanks?( you| a lot)?|thank you|"
    r"ok(ay)?|cool|great|nice|bye|yes|no|nope|nahi|haan|hmm+|\?)[\s!.?]*", re.I)


class SearchFirst(AgentMiddleware):
    """Every real question starts with a website search. Told to search in
    the prompt alone, the model skips it once the conversation or the example
    chats seem to hold the answer -- and then repeats facts the website may
    since have changed. Only the first model call of each visitor message is
    forced; after the results it answers freely."""

    @staticmethod
    def _force(request):
        last = request.messages[-1] if request.messages else None
        if isinstance(last, HumanMessage) and not SMALL_TALK_RE.fullmatch(last.text.strip()):
            return request.override(tool_choice="any")
        return request

    def wrap_model_call(self, request, handler):
        return handler(self._force(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._force(request))


def build_agent(collection, model=None):
    """The agent over one index. `model` replaces the configured chat model (tests)."""

    @tool
    async def search_website(query: str) -> str:
        """Search IDCUBE's website content (pages, FAQs, brochures and other PDFs)
        and return the most relevant excerpts, each with its URL. Write the query
        in English."""
        chunks = await asyncio.to_thread(retrieve, query.strip()[:MAX_SEARCH_QUERY_CHARS], collection)
        return build_context(chunks) or "No matching content."

    return create_agent(
        model or _chat_model(),
        [search_website],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            SearchFirst(),
            # Caches the instructions, the tool and the conversation so far; a
            # no-op for non-Anthropic models.
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            # A third search is refused, so the model answers from what it has.
            ToolCallLimitMiddleware(tool_name="search_website", run_limit=MAX_SEARCHES),
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
            ModelRefusalMiddleware(),
        ],
    )


def _add_usage(usage: dict, msg: AIMessage):
    # LangChain's input_tokens includes cached tokens; split them back out so
    # the log prices each kind correctly.
    u = msg.usage_metadata or {}
    details = u.get("input_token_details") or {}
    read = details.get("cache_read") or 0
    write = sum(details.get(k) or 0 for k in
                ("cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"))
    for key, value in (("input_tokens", max(u.get("input_tokens", 0) - read - write, 0)),
                       ("output_tokens", u.get("output_tokens", 0)),
                       ("cache_read_input_tokens", read),
                       ("cache_creation_input_tokens", write),
                       ("rounds", 1)):
        usage[key] = usage.get(key, 0) + value


async def stream_reply(agent, messages: list, usage: dict):
    """Run the agent on a conversation. Yields ("delta", text) as the reply is
    written, ("keepalive", None) while searching, and finally ("final", text)
    -- the last model turn's text only, so a stray sentence before a search
    never ends up in the reply. A refusal raises ModelRefusalError."""
    final = ""
    async for mode, data in agent.astream({"messages": messages}, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, meta = data
            # Tool results stream here too; only the model's own text goes out.
            if meta.get("langgraph_node") == "model" and chunk.text:
                yield "delta", chunk.text
            continue
        for msg in (data.get("model") or {}).get("messages", []):
            if not isinstance(msg, AIMessage):
                continue
            _add_usage(usage, msg)
            if msg.tool_calls:
                usage["searches"] = usage.get("searches", 0) + len(msg.tool_calls)
                yield "keepalive", None
            else:
                final = msg.text
    yield "final", final.strip()
