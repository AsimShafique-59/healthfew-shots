import os
import json
import socket
import threading
import time
import requests
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st
import uvicorn

# ------------------ ENV ------------------
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError("❌ GROQ_API_KEY not found in .env")
os.environ["GROQ_API_KEY"] = groq_key

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
BACKEND_URL = f"http://{API_HOST}:{API_PORT}"

def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

# ------------------ FASTAPI APP ------------------
app = FastAPI(title="Healthcare AI Chat API", version="1.2")

SYSTEM_PROMPT = (
    "You are a friendly and professional AI healthcare assistant. "
    "Always greet the user politely first (e.g., 'Hello! 👋') before answering. "
    "Answer clearly and helpfully if the question is healthcare-related, including definitions when appropriate. "
    "Only provide answers related to medical topics, diseases, symptoms, treatments, "
    "wellness, nutrition, or healthcare systems. "
    "If the user's question is not related to healthcare, reply exactly: "
    "\"Sorry, I can only answer healthcare-related questions.\""
)

FEW_SHOTS = [
    {
        "user": "What is healthcare?",
        "ai": "Hello! 👋 Healthcare is the organized provision of medical services to individuals or communities, "
              "aimed at maintaining or improving health through prevention, diagnosis, treatment, and rehabilitation."
    },
    {
        "user": "I have a sore throat for two days",
        "ai": "Hello! 👋 Rest, fluids, and warm salt-water gargles can help. Consider acetaminophen or ibuprofen for pain if suitable. "
              "Seek care if you develop high fever, difficulty breathing, or symptoms persist beyond 3–5 days."
    },
    {
        "user": "I'm starting a new workout routine",
        "ai": "Hello! 👋 Great! Warm up 5–10 minutes, progress gradually, prioritize form, hydrate, and include rest days to prevent injury."
    },
    {
        "user": "Who will win the next football match?",
        "ai": "Hello! 👋 Sorry, I can only answer healthcare-related questions."
    }
]

REFINER_PROMPT_TEMPLATE = """
You are a query *refiner* for a healthcare-only assistant.

Goal: Rewrite the latest user question into ONE standalone, precise healthcare query optimized for retrieval/answering.

Rules:
- If the latest question is completely unrelated to health, medicine, wellness, or healthcare systems, output exactly: SORRY_OFF_TOPIC
- Otherwise, keep only the healthcare content.
- Remove pronouns/ambiguity; include relevant clinical terms, demographics, timeframe, and units if present.
- Be specific but brief (<= 1 sentence). Do NOT answer the question.

Chat history:
{history}

Latest user question:
{question}

Refined query:
"""

# ---------- Models ----------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    history: List[Message]
    user_input: str
    model: str = "llama-3.1-8b-instant"  # ensure your Groq key has access
    temperature: float = 0.2
    use_refine: bool = True

class ChatResponse(BaseModel):
    ai_response: str
    refined_query: Optional[str] = None
    off_topic: bool = False

# ---------- Helpers ----------
def build_prompt(history: List[dict], last_user_override: Optional[str] = None) -> str:
    lines = [f"System: {SYSTEM_PROMPT}"]
    for i, ex in enumerate(FEW_SHOTS, start=1):
        lines.append(f"\nExample {i}:\nUser: {ex['user']}\nAI: {ex['ai']}")
    lines.append("\nConversation so far:")

    hist_copy = history[:]
    if last_user_override:
        for idx in range(len(hist_copy) - 1, -1, -1):
            if hist_copy[idx]["role"].lower() == "user":
                hist_copy[idx] = {"role": "user", "content": last_user_override}
                break

    for m in hist_copy:
        if m["role"].lower() != "system":
            lines.append(f"{m['role'].capitalize()}: {m['content']}")
    lines.append("AI:")
    return "\n".join(lines)

def get_text_history_for_refiner(history: List[dict]) -> str:
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history][-12:])

HEALTH_KEYWORDS = {
    "health", "healthcare", "doctor", "hospital", "medicine", "medical", "symptom",
    "disease", "treatment", "therapy", "diagnosis", "nutrition", "wellness", "vaccine"
}

def likely_health_related(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in HEALTH_KEYWORDS)

def refine_query(history: List[dict], user_question: str, model_name: str):
    """Refine the query, but allow obvious healthcare questions through."""
    try:
        if likely_health_related(user_question):
            return user_question, False, None

        refiner = ChatGroq(model_name=model_name, temperature=0.0)
        prompt = REFINER_PROMPT_TEMPLATE.format(
            history=get_text_history_for_refiner(history),
            question=user_question
        )
        out = refiner.invoke(prompt)
        refined = getattr(out, "content", "").strip() if out else ""
        if refined == "SORRY_OFF_TOPIC":
            return "", True, None
        if not refined or len(refined.split()) < 3 or len(refined) > 400:
            return user_question, False, None
        return refined, False, None
    except Exception as e:
        return user_question, False, str(e)

# ---------- Health check ----------
@app.get("/")
def home():
    return {"status": "Healthcare AI Chat API is running"}

# ---------- Non-streaming chat (kept for compatibility) ----------
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    refined_text = req.user_input
    off_topic = False

    if req.use_refine:
        refined_text, off_topic, _ = refine_query([m.dict() for m in req.history], req.user_input, req.model)

    if off_topic:
        return ChatResponse(ai_response='Hello! 👋 Sorry, I can only answer healthcare-related questions.', refined_query=None, off_topic=True)

    prompt = build_prompt([m.dict() for m in req.history], last_user_override=refined_text)
    llm = ChatGroq(model_name=req.model, temperature=req.temperature)
    out = llm.invoke(prompt)
    return ChatResponse(ai_response=getattr(out, "content", "❌ No response"), refined_query=refined_text)

# ---------- Streaming chat (sends meta as first line) ----------
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    refined_text = req.user_input
    off_topic = False
    if req.use_refine:
        refined_text, off_topic, _ = refine_query([m.dict() for m in req.history], req.user_input, req.model)

    def _gen():
        # 1) Send metadata line first (no headers, keep full Unicode)
        meta = {"refined": refined_text, "off_topic": off_topic}
        yield "__META__:" + json.dumps(meta, ensure_ascii=True) + "\n"

        # 2) Then stream the content
        if off_topic:
            yield "Hello! 👋 Sorry, I can only answer healthcare-related questions."
            return

        prompt = build_prompt([m.dict() for m in req.history], last_user_override=refined_text)
        llm = ChatGroq(model_name=req.model, temperature=req.temperature)
        for chunk in llm.stream(prompt):
            token = getattr(chunk, "content", "")
            if token:
                yield token

    return StreamingResponse(_gen(), media_type="text/plain")

# ------------------ BACKGROUND FASTAPI SERVER (auto-skip if port busy) ------------------
def run_api():
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")

if not is_port_in_use(API_HOST, API_PORT):
    threading.Thread(target=run_api, daemon=True).start()
    time.sleep(1)  # give uvicorn a moment
else:
    print(f"ℹ️ API already running at {BACKEND_URL}; will not start another instance.")

# ------------------ STREAMLIT FRONTEND ------------------
st.set_page_config(page_title="Healthcare AI Chat", layout="wide")
st.title("🩺 Healthcare AI Chat")

st.sidebar.header("⚙️ Settings")
use_refine = st.sidebar.checkbox("Refine queries", value=True)
show_refined = st.sidebar.checkbox("Show refined query", value=True)
model_name = st.sidebar.text_input("Groq model", value="llama-3.1-8b-instant")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
backend_override = st.sidebar.text_input("Backend URL", value=BACKEND_URL)

if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": "start"}]

# Render chat history
for msg in st.session_state.history:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_input = st.chat_input("Ask a health-related question...")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    payload = {
        "history": st.session_state.history,
        "user_input": user_input,
        "model": model_name,
        "temperature": float(temperature),
        "use_refine": use_refine
    }

    try:
        # Start streaming request
        resp = requests.post(f"{backend_override}/chat/stream", json=payload, stream=True, timeout=300)
        resp.raise_for_status()

        # Read the first line for metadata
        lines = resp.iter_lines(decode_unicode=True)
        first = next(lines, "")
        refined_q = None
        off_topic = False
        if isinstance(first, bytes):
            first = first.decode("utf-8", errors="ignore")

        if first.startswith("__META__:"):
            try:
                meta_json = first.split(":", 1)[1]
                meta = json.loads(meta_json)
                refined_q = meta.get("refined")
                off_topic = bool(meta.get("off_topic"))
            except Exception:
                pass
        else:
            # If no meta, treat the first as content
            # We'll print it along with the rest of the stream
            pass

        # Show refined query (optional)
        if use_refine and show_refined and refined_q and not off_topic and refined_q != user_input:
            st.markdown(
                f"<div style='font-size:0.85rem; color:#666;'>Refined query: <em>{refined_q}</em></div>",
                unsafe_allow_html=True
            )

        # Stream remaining lines (plus the first if it wasn't meta)
        with st.chat_message("assistant"):
            def _line_iter():
                # If the first line wasn't meta, yield it
                if first and not first.startswith("__META__:"):
                    yield first + "\n"
                for line in lines:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="ignore")
                    yield line + "\n"

            full_text = st.write_stream(_line_iter)

        st.session_state.history.append({"role": "assistant", "content": full_text or ""})

    except Exception as e:
        err = f"❌ Error: {e}"
        with st.chat_message("assistant"):
            st.markdown(err)
        st.session_state.history.append({"role": "assistant", "content": err})
