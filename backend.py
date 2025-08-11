# backend.py

import os
import json
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from prompts import SYSTEM_PROMPT, FEW_SHOTS, REFINER_PROMPT_TEMPLATE

# ---------- ENV ----------
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError("❌ GROQ_API_KEY not found in .env")
os.environ["GROQ_API_KEY"] = groq_key

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ---------- FASTAPI ----------
app = FastAPI(title="Healthcare AI Chat API", version="2.0")

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

    hist_copy = [dict(m) for m in history]
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
# refining qerry
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

# ---------- Non-streaming chat ----------
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
        # 1) Send metadata line first (no headers, keeps full Unicode)
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

# ---------- Main ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
