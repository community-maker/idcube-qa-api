/*
 * IDCUBE website chat widget.
 *
 * Embed with one tag (e.g. in the WordPress footer):
 *   <script src="https://idcube-qa-api.onrender.com/widget.js"
 *           data-api="https://idcube-qa-api.onrender.com" defer></script>
 *
 * Optional attributes: data-title, data-color, data-greeting, data-privacy-url.
 *
 * Talks only to the relay in chat_proxy.py -- never to Purple Fabric directly,
 * so no credentials ever reach the browser. Renders inside a Shadow DOM so the
 * site theme's CSS can't break it (and it can't break the theme). Agent text is
 * built with DOM APIs, never innerHTML, so model output can't inject markup.
 */
(() => {
  "use strict";
  if (window.__idcubeChatLoaded) return;
  window.__idcubeChatLoaded = true;

  const script =
    document.currentScript ||
    Array.from(document.scripts).find((s) => /\/widget\.js(\?|$)/.test(s.src));
  const ds = (script && script.dataset) || {};
  const cfg = {
    api: (ds.api || (script ? new URL(script.src).origin : "")).replace(/\/+$/, ""),
    title: ds.title || "IDCUBE Assistant",
    color: ds.color || "#0b5cad",
    greeting:
      ds.greeting ||
      "Hi! I'm the IDCUBE assistant. Ask me about our products, solutions, partners or support.",
    privacyUrl: ds.privacyUrl || "https://www.idcubesystems.com/privacy-policy/",
  };
  const FALLBACK =
    "Sorry, I'm having trouble answering right now. Please try again in a moment, or reach our team at contact@idcubesystems.com.";
  const STORE_KEY = "idcube-chat-v1";
  const MAX_STORED = 40;
  const MAX_CHARS = 1000;

  // ---- state (per browser tab, survives page navigation on the site) ----
  const state = { session: null, open: false, messages: [] };
  try {
    Object.assign(state, JSON.parse(sessionStorage.getItem(STORE_KEY) || "{}"));
  } catch (_) {}
  if (!state.messages.length) state.messages.push({ role: "assistant", text: cfg.greeting });
  const save = () => {
    state.messages = state.messages.slice(-MAX_STORED);
    try {
      sessionStorage.setItem(STORE_KEY, JSON.stringify(state));
    } catch (_) {}
  };
  let busy = false;

  // ---- DOM ----
  const host = document.createElement("div");
  host.id = "idcube-chat-widget";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `
    <style>
      :host { all: initial; }
      * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
      .launcher { position: fixed; right: 20px; bottom: 20px; z-index: 2147483000; width: 60px; height: 60px;
        border-radius: 50%; border: 0; cursor: pointer; background: var(--brand); color: #fff;
        box-shadow: 0 6px 20px rgba(0,0,0,.25); display: flex; align-items: center; justify-content: center; }
      .launcher:focus-visible, button:focus-visible, textarea:focus-visible, a:focus-visible { outline: 3px solid #ffbf47; outline-offset: 2px; }
      .launcher svg { width: 28px; height: 28px; }
      .panel { position: fixed; right: 20px; bottom: 92px; z-index: 2147483000; width: 380px; height: min(600px, calc(100vh - 120px));
        background: #fff; color: #1d1d1f; border-radius: 14px; box-shadow: 0 12px 40px rgba(0,0,0,.25);
        display: flex; flex-direction: column; overflow: hidden; font-size: 14px; line-height: 1.5; }
      .panel[hidden] { display: none; }
      .head { background: var(--brand); color: #fff; padding: 14px 16px; display: flex; align-items: center; justify-content: space-between; }
      .head h2 { margin: 0; font-size: 16px; font-weight: 600; }
      .close { background: transparent; border: 0; color: #fff; font-size: 22px; line-height: 1; cursor: pointer; padding: 4px 8px; border-radius: 6px; }
      .log { flex: 1; overflow-y: auto; padding: 14px; background: #f8fafc; display: flex; flex-direction: column; gap: 10px; }
      .msg { max-width: 85%; padding: 10px 12px; border-radius: 12px; word-wrap: break-word; overflow-wrap: anywhere; }
      .msg.user { align-self: flex-end; background: var(--brand); color: #fff; border-bottom-right-radius: 4px; white-space: pre-wrap; }
      .msg.assistant { align-self: flex-start; background: #fff; border: 1px solid #e2e8f0; border-bottom-left-radius: 4px; }
      .msg p { margin: 0 0 8px; } .msg p:last-child { margin-bottom: 0; }
      .msg ul, .msg ol { margin: 4px 0 8px; padding-left: 20px; }
      .msg a { color: var(--brand); text-decoration: underline; }
      .msg code { background: #f1f5f9; padding: 1px 4px; border-radius: 4px; font-family: Consolas, monospace; font-size: 13px; }
      .typing { display: inline-flex; gap: 4px; padding: 2px 0; }
      .typing span { width: 7px; height: 7px; border-radius: 50%; background: #94a3b8; animation: blink 1.2s infinite both; }
      .typing span:nth-child(2) { animation-delay: .2s; } .typing span:nth-child(3) { animation-delay: .4s; }
      @keyframes blink { 0%, 80%, 100% { opacity: .3; } 40% { opacity: 1; } }
      .hint { font-size: 12px; color: #64748b; margin-top: 4px; }
      .starters { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 14px 10px; background: #f8fafc; }
      .starters button { background: #fff; border: 1px solid var(--brand); color: var(--brand); border-radius: 999px;
        padding: 6px 10px; font-size: 13px; cursor: pointer; text-align: left; }
      form { display: flex; gap: 8px; padding: 10px; border-top: 1px solid #e2e8f0; background: #fff; }
      textarea { flex: 1; resize: none; border: 1px solid #cbd5e1; border-radius: 10px; padding: 9px 10px; font-size: 14px;
        max-height: 110px; min-height: 40px; color: #1d1d1f; background: #fff; }
      .send { border: 0; border-radius: 10px; background: var(--brand); color: #fff; padding: 0 14px; font-size: 14px; cursor: pointer; }
      .send:disabled { opacity: .5; cursor: default; }
      .foot { font-size: 11px; color: #64748b; padding: 0 12px 8px; background: #fff; }
      .foot a { color: #64748b; }
      @media (max-width: 480px) {
        .panel { right: 8px; left: 8px; bottom: 84px; width: auto; height: calc(100vh - 100px); }
        textarea { font-size: 16px; } /* below 16px iOS Safari zooms the page on focus */
      }
      @media (prefers-reduced-motion: reduce) { .typing span { animation: none; } }
    </style>
    <button class="launcher" type="button" aria-label="Open chat" aria-expanded="false">
      <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M4 4h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8l-4 4V6a2 2 0 0 1 2-2z"/></svg>
    </button>
    <div class="panel" role="dialog" aria-labelledby="idc-title" hidden>
      <div class="head"><h2 id="idc-title"></h2><button class="close" type="button" aria-label="Close chat">&times;</button></div>
      <div class="log" role="log" aria-live="polite"></div>
      <div class="starters"></div>
      <form>
        <label for="idc-input" style="position:absolute;left:-9999px">Type your message</label>
        <textarea id="idc-input" rows="1" placeholder="Type your question..." maxlength="${MAX_CHARS}"></textarea>
        <button class="send" type="submit">Send</button>
      </form>
      <div class="foot">AI assistant &mdash; answers may be incomplete. <a target="_blank" rel="noopener noreferrer">Privacy policy</a></div>
    </div>`;

  const $ = (sel) => root.querySelector(sel);
  const launcher = $(".launcher");
  const panel = $(".panel");
  const log = $(".log");
  const starters = $(".starters");
  const form = $("form");
  const input = $("textarea");
  const sendBtn = $(".send");
  root.host.style.setProperty("--brand", cfg.color);
  $("#idc-title").textContent = cfg.title;
  $(".foot a").href = cfg.privacyUrl;

  // ---- safe markdown -> DOM (no innerHTML for model output) ----
  const INLINE_RE =
    /(\*\*[^*\n]+\*\*|`[^`\n]+`|\[[^\]\n]+\]\((?:https?:\/\/|mailto:|tel:)[^)\s]+\)|(?:https?:\/\/|www\.)[^\s<>()]*[^\s<>().,;:!?'"]|[\w.+-]+@[\w-]+(?:\.[\w-]+)+)/g;

  function makeLink(href, text) {
    if (!/^(https?:|mailto:|tel:)/i.test(href)) return document.createTextNode(text);
    const a = document.createElement("a");
    a.href = href;
    a.textContent = text;
    let sameSite = false;
    try {
      const u = new URL(href);
      sameSite = /(^|\.)idcubesystems\.com$/i.test(u.hostname) && /(^|\.)idcubesystems\.com$/i.test(location.hostname);
    } catch (_) {}
    if (!sameSite && /^https?:/i.test(href)) {
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    }
    return a;
  }

  function renderInline(text, parent) {
    let last = 0;
    for (const m of text.matchAll(INLINE_RE)) {
      if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
      const t = m[0];
      if (t.startsWith("**")) {
        const b = document.createElement("strong");
        b.textContent = t.slice(2, -2);
        parent.appendChild(b);
      } else if (t.startsWith("`")) {
        const c = document.createElement("code");
        c.textContent = t.slice(1, -1);
        parent.appendChild(c);
      } else if (t.startsWith("[")) {
        const [, label, href] = t.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
        parent.appendChild(makeLink(href, label));
      } else if (t.includes("@") && !/^(https?:\/\/|www\.)/i.test(t)) {
        parent.appendChild(makeLink("mailto:" + t, t));
      } else {
        parent.appendChild(makeLink(t.startsWith("www.") ? "https://" + t : t, t));
      }
      last = m.index + t.length;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
  }

  function renderMarkdown(md, el) {
    el.textContent = "";
    let para = null;
    let list = null;
    const flush = () => {
      para = null;
      list = null;
    };
    for (const raw of md.replace(/\r/g, "").split("\n")) {
      const line = raw.trimEnd();
      const heading = line.match(/^#{1,6}\s+(.*)$/);
      const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
      const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (!line.trim()) {
        flush();
      } else if (heading) {
        flush();
        const p = document.createElement("p");
        const s = document.createElement("strong");
        renderInline(heading[1], s);
        p.appendChild(s);
        el.appendChild(p);
      } else if (bullet || numbered) {
        const tag = bullet ? "UL" : "OL";
        if (!list || list.tagName !== tag) {
          para = null;
          list = document.createElement(tag.toLowerCase());
          el.appendChild(list);
        }
        const li = document.createElement("li");
        renderInline((bullet || numbered)[1], li);
        list.appendChild(li);
      } else {
        list = null;
        if (!para) {
          para = document.createElement("p");
          el.appendChild(para);
        } else {
          para.appendChild(document.createElement("br"));
        }
        renderInline(line, para);
      }
    }
  }

  // ---- messages ----
  function addBubble(role, text) {
    const div = document.createElement("div");
    div.className = "msg " + role;
    if (role === "user") div.textContent = text;
    else renderMarkdown(text, div);
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return div;
  }

  function showTyping(bubble) {
    bubble.textContent = "";
    const t = document.createElement("div");
    t.className = "typing";
    t.setAttribute("aria-label", "Assistant is typing");
    t.append(document.createElement("span"), document.createElement("span"), document.createElement("span"));
    bubble.appendChild(t);
    const hint = document.createElement("div");
    hint.className = "hint";
    bubble.appendChild(hint);
    // The relay sleeps when idle; the first reply after a quiet spell is slow.
    return setTimeout(() => (hint.textContent = "Waking up the assistant, this can take a few seconds..."), 8000);
  }

  // ---- network ----
  async function ensureSession() {
    if (state.session) return state.session;
    const r = await fetch(cfg.api + "/chat/session", { method: "POST" });
    if (!r.ok) throw new Error(FALLBACK);
    state.session = (await r.json()).session;
    save();
    return state.session;
  }

  const postMessage = (query) =>
    fetch(cfg.api + "/chat/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: state.session, query }),
    });

  async function send(text) {
    text = text.trim().slice(0, MAX_CHARS);
    if (busy || !text) return;
    busy = true;
    sendBtn.disabled = true;
    starters.textContent = "";
    input.value = "";
    autosize();
    state.messages.push({ role: "user", text });
    addBubble("user", text);
    const bubble = addBubble("assistant", "");
    const hintTimer = showTyping(bubble);

    let acc = "";
    let finalText = null;
    let failure = null;
    let frame = 0;
    const paint = () => {
      frame = 0;
      renderMarkdown(finalText || acc, bubble);
      log.scrollTop = log.scrollHeight;
    };
    try {
      await ensureSession();
      let resp = await postMessage(text);
      if (resp.status === 401) {
        // Session token no longer valid (e.g. the relay restarted) -- start fresh.
        state.session = null;
        await ensureSession();
        resp = await postMessage(text);
      }
      if (resp.status === 429) throw new Error("You're sending messages quickly. Please wait a minute and try again.");
      if (!resp.ok || !resp.body) throw new Error(FALLBACK);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let done = false;
      while (!done) {
        const { value, done: streamDone } = await reader.read();
        if (streamDone) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const event = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          for (const line of event.split("\n")) {
            if (!line.startsWith("data:")) continue;
            let msg;
            try {
              msg = JSON.parse(line.slice(5).trim());
            } catch (_) {
              continue;
            }
            if (msg.type === "delta") acc += msg.text;
            else if (msg.type === "final") finalText = msg.text;
            else if (msg.type === "error") failure = msg.text || FALLBACK;
            else if (msg.type === "end") done = true;
            if (failure) done = true;
          }
        }
        if ((acc || finalText) && !frame && !failure) {
          clearTimeout(hintTimer);
          frame = requestAnimationFrame(paint);
        }
      }
    } catch (e) {
      failure = (e && e.message) || FALLBACK;
    }
    clearTimeout(hintTimer);
    if (frame) cancelAnimationFrame(frame);
    const reply = failure || finalText || acc || FALLBACK;
    renderMarkdown(reply, bubble);
    log.scrollTop = log.scrollHeight;
    state.messages.push({ role: "assistant", text: reply });
    save();
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }

  async function loadStarters() {
    if (state.messages.some((m) => m.role === "user")) return;
    try {
      const r = await fetch(cfg.api + "/chat/starters");
      if (!r.ok) return;
      const { starters: list } = await r.json();
      starters.textContent = "";
      (list || []).slice(0, 4).forEach((q) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = q;
        b.addEventListener("click", () => send(q));
        starters.appendChild(b);
      });
    } catch (_) {}
  }

  // ---- open / close ----
  function setOpen(open) {
    state.open = open;
    save();
    panel.hidden = !open;
    launcher.setAttribute("aria-expanded", String(open));
    launcher.setAttribute("aria-label", open ? "Close chat" : "Open chat");
    if (open) {
      log.scrollTop = log.scrollHeight;
      input.focus();
      loadStarters();
    }
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 110) + "px";
  }

  launcher.addEventListener("click", () => setOpen(panel.hidden));
  $(".close").addEventListener("click", () => {
    setOpen(false);
    launcher.focus();
  });
  panel.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      setOpen(false);
      launcher.focus();
    }
  });
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    send(input.value);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input.value);
    }
  });
  input.addEventListener("input", autosize);

  state.messages.forEach((m) => addBubble(m.role, m.text));
  const mount = () => {
    document.body.appendChild(host);
    if (state.open) setOpen(true);
  };
  if (document.body) mount();
  else document.addEventListener("DOMContentLoaded", mount);
})();
