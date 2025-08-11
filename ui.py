# ui.py

import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---- Fixed config from env (no model/backend fields in UI) ----
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_TEMP = float(os.getenv("TEMPERATURE", "0.2"))

st.set_page_config(page_title="Healthcare AI Chat (Groq)", layout="wide", initial_sidebar_state="expanded")

# ---- Styles ----
st.markdown(
    """
<style>
:root {
  --bg:#ffffff; --fg:#111111; --muted:#6b7280; --chip-bg:#f1f5f9; --chip-fg:#0f172a;
  --border:#e5e7eb; --shadow:0 2px 14px rgba(0,0,0,.04);
}
body, .stApp { background:var(--bg); color:var(--fg); }
.block-container { padding-top: 1rem; }
.chat-header { display:flex; align-items:center; gap:.6rem; margin-bottom:.6rem; }
.app-title { font-size:1.35rem; font-weight:700; }
.chip { display:inline-flex; align-items:center; gap:.35rem; background:var(--chip-bg); color:var(--chip-fg);
  padding:.25rem .6rem; border-radius:999px; font-size:.82rem; border:1px solid var(--border); }
hr.sep { border:none; border-top:1px solid var(--border); margin:.6rem 0 1rem 0; }
.typing { font-size:.9rem; color:var(--muted); }
.hero {
  max-width: 800px; margin: 4vh auto 2vh auto; padding: 18px 22px;
  border: 1px solid var(--border); border-radius: 16px; box-shadow: var(--shadow);
  background: #fff;
}
.hero h3 { margin: 0 0 6px 0; }
.hero ul { margin: 6px 0 0 1.1rem; }
.center-wrap { max-width: 900px; margin: 0 auto; }
.room-title { font-size:.9rem; color:var(--muted); margin-top:-.4rem; }
</style>
""",
    unsafe_allow_html=True
)

# =========================
# Chat Rooms State Helpers
# =========================
def _init_rooms():
    if "chats" not in st.session_state:
        st.session_state.chats = {
            "Chat 1": [{"role": "system", "content": "start"}]
        }
    if "current_room" not in st.session_state:
        st.session_state.current_room = "Chat 1"
    if "room_counter" not in st.session_state:
        st.session_state.room_counter = 1
    if "room_titles" not in st.session_state:
        st.session_state.room_titles = {"Chat 1": "Chat 1"}

def current_history():
    return st.session_state.chats[st.session_state.current_room]

def set_history(hist):
    st.session_state.chats[st.session_state.current_room] = hist

def create_room():
    st.session_state.room_counter += 1
    name = f"Chat {st.session_state.room_counter}"
    st.session_state.chats[name] = [{"role": "system", "content": "start"}]
    st.session_state.room_titles[name] = name
    st.session_state.current_room = name

def delete_current_room():
    if len(st.session_state.chats) <= 1:
        return  # don't delete the last room
    room = st.session_state.current_room
    # choose another room (first remaining)
    remaining = [r for r in st.session_state.chats.keys() if r != room]
    st.session_state.current_room = remaining[0]
    st.session_state.chats.pop(room, None)
    st.session_state.room_titles.pop(room, None)

def rename_room_on_first_user_msg(room, text):
    # Update room title to first user message (truncated) one time
    title = st.session_state.room_titles.get(room, room)
    if title.startswith("Chat "):  # only auto-rename default titles
        snippet = text.strip().replace("\n", " ")
        if len(snippet) > 32:
            snippet = snippet[:32] + "…"
        st.session_state.room_titles[room] = snippet or title

# =========================
# Sidebar
# =========================
_init_rooms()

st.sidebar.header("💬 Chats")
# Room switcher
room_names = list(st.session_state.chats.keys())
display_names = [st.session_state.room_titles.get(r, r) for r in room_names]
selected = st.sidebar.radio(
    "Chat rooms", room_names, index=room_names.index(st.session_state.current_room), format_func=lambda r: st.session_state.room_titles.get(r, r)
)

if selected != st.session_state.current_room:
    st.session_state.current_room = selected
    st.rerun()

colA, colB, colC = st.sidebar.columns([1,1,1.2])
with colA:
    if st.button("🆕 New"):
        create_room()
        st.rerun()
with colB:
    if st.button("🗑️ Delete"):
        delete_current_room()
        st.rerun()
with colC:
    if st.button("🔌 Health"):
        try:
            r = requests.get(f"{BACKEND_URL}/")
            ok = r.ok and r.json().get("status", "").lower().find("running") >= 0
            st.sidebar.success("Backend OK ✅" if ok else "Backend responded")
        except Exception as e:
            st.sidebar.error(f"Backend error: {e}")

st.sidebar.header("⚙️ Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, DEFAULT_TEMP, 0.1)

# =========================
# Header
# =========================
st.markdown(
    f"""
<div class="center-wrap">
  <div class="chat-header">
    <div class="app-title">🩺 Healthcare AI Chat</div>
    <div class="chip" title="Answers only on healthcare topics">Healthcare-only</div>
  </div>
  <div class="room-title">Room: <strong>{st.session_state.room_titles.get(st.session_state.current_room, st.session_state.current_room)}</strong></div>
  <hr class="sep" />
</div>
""",
    unsafe_allow_html=True
)

# =========================
# Render chat or welcome
# =========================
hist = current_history()

if len(hist) == 1 and hist[0]["role"] == "system":
    st.markdown(
        """
<div class="center-wrap">
  <div class="hero">
    <h3>👋 Welcome!</h3>
    <p>This is a new chat room. Ask anything related to <strong>healthcare</strong>: medical topics, symptoms, treatments, wellness, or healthcare systems.</p>
    <ul>
      <li>What is healthcare?</li>
      <li>I have a sore throat for two days, what should I do?</li>
      <li>How much water should I drink daily?</li>
    </ul>
  </div>
</div>
""",
        unsafe_allow_html=True
    )
else:
    for msg in hist:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# =========================
# Input + Streaming
# =========================
user_input = st.chat_input("Ask a health-related question…")
if user_input:
    # Add user message to this room's history
    hist.append({"role": "user", "content": user_input})
    set_history(hist)
    rename_room_on_first_user_msg(st.session_state.current_room, user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    payload = {
        "history": hist,
        "user_input": user_input,
        "model": MODEL_NAME,
        "temperature": float(temperature),
        "use_refine": True,  # backend feature stays on
    }

    with st.chat_message("assistant"):
        typing_ph = st.empty()
        typing_ph.markdown('<div class="typing">Assistant is typing…</div>', unsafe_allow_html=True)

        try:
            resp = requests.post(f"{BACKEND_URL}/chat/stream", json=payload, stream=True, timeout=300)
            resp.raise_for_status()

            # First line is meta (__META__:{...})
            lines = resp.iter_lines(decode_unicode=True)
            first = next(lines, "")
            if isinstance(first, bytes):
                first = first.decode("utf-8", errors="ignore")

            def _line_iter(first_line=first, lines_iter=lines):
                emitted_any = False
                if first_line and not (isinstance(first_line, str) and first_line.startswith("__META__:")):
                    emitted_any = True
                    yield first_line + "\n"
                for line in lines_iter:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="ignore")
                    emitted_any = True
                    yield line + "\n"
                if not emitted_any:
                    yield ""

            typing_ph.empty()
            full_text = st.write_stream(_line_iter)

            # Save assistant message to this room
            hist.append({"role": "assistant", "content": full_text or ""})
            set_history(hist)

        except Exception as e:
            typing_ph.empty()
            err = f"❌ Error: {e}"
            st.markdown(err)
            hist.append({"role": "assistant", "content": err})
            set_history(hist)
