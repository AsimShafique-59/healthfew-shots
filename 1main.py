import time
import uuid
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ---------------- ENV ----------------
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("❌ GROQ_API_KEY not found. Add it to your .env file.")
    st.stop()
os.environ["GROQ_API_KEY"] = groq_key  # ensure SDK sees it

# ---------------- UI (white theme) ----------------
st.set_page_config(page_title="Healthcare AI Chat (Groq)", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
        body { background-color: white; color: black; }
        .stChatMessage { max-width: 900px; margin-left: auto; margin-right: auto; }
        .stChatFloatingInputContainer { max-width: 900px; margin-left: auto; margin-right: auto; }
        .refined { font-size: 0.85rem; color: #666; margin-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- System Prompt ----------------
SYSTEM_PROMPT = (
    "You are a helpful AI assistant that only answers health-related questions. "
    "Only provide answers related to medical topics, diseases, symptoms, treatments, "
    "wellness, nutrition, or healthcare systems. If the user's question is not "
    "related to healthcare, reply exactly: \"Sorry, I can only answer healthcare-related questions.\""
)

# ---------------- 3-shot Few-shot Examples ----------------
FEW_SHOTS = [
    {
        "user": "first greet the user beautifully. I have a sore throat and mild fever for two days. What can I do at home, and when should I see a doctor?",
        "ai": (
            "A sore throat with mild fever is often viral. At home: rest, drink fluids, warm salt-water gargles, "
            "use acetaminophen or ibuprofen as directed, and consider honey or lozenges. Seek care urgently if you "
            "have trouble breathing, drooling, a stiff neck, severe dehydration, a fever >39°C, symptoms >5–7 days, "
            "or if you’re immunocompromised. This is general info, not a diagnosis."
        ),
    },
    {
        "user": "I'm starting a new workout routine. Any tips to avoid injury?",
        "ai": (
            "Increase intensity gradually (10% per week), warm up 5–10 minutes, use proper form, and schedule rest days. "
            "Hydrate, sleep 7–9 hours, and consider cross-training. If pain is sharp, swelling appears, or pain persists "
            "beyond 48–72 hours, pause training and seek medical advice."
        ),
    },
    {
        "user": "Who will win the next football match?",
        "ai": 'Sorry, I can only answer healthcare-related questions.',
    },
]

# ---------------- Helpers ----------------
def new_chat_name(n: int) -> str:
    ts = time.strftime("%H:%M")
    return f"Chat {n} • {ts}"

def create_empty_chat(name: str):
    return {
        "name": name,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
    }

def build_prompt_from_history(history, last_user_override: str | None = None):
    """
    Build a prompt string from history.
    If last_user_override is provided, we replace the content of the *last* user message
    in the constructed prompt with that override (without changing session_state).
    """
    lines = [f"System: {SYSTEM_PROMPT}"]

    # Inject 3-shot examples
    lines.append("\nHere are examples of the style and scope you must follow:")
    for i, ex in enumerate(FEW_SHOTS, start=1):
        lines.append(f"\nExample {i}:")
        lines.append(f"User: {ex['user']}")
        lines.append(f"AI: {ex['ai']}")

    # Then the live conversation
    lines.append("\nConversation so far:")
    # Find index of last user message
    last_user_index = max((i for i, m in enumerate(history) if m["role"] == "user"), default=-1)

    for idx, m in enumerate(history):
        if m["role"] == "user":
            content = m["content"]
            if last_user_override is not None and idx == last_user_index:
                content = last_user_override
            lines.append(f"User: {content}")
        elif m["role"] == "assistant":
            lines.append(f"AI: {m['content']}")

    lines.append("AI:")
    return "\n".join(lines)

# --- Query Refinement ---
REFINER_PROMPT_TEMPLATE = """
You are a query *refiner* for a healthcare-only assistant.

Goal: Rewrite the latest user question into ONE standalone, precise healthcare query optimized for retrieval/answering.

Rules:
- Keep only healthcare content. If the latest question is NOT about healthcare, output exactly: SORRY_OFF_TOPIC
- Remove pronouns/ambiguity; include relevant clinical terms, demographics, timeframe, and units if present.
- Be specific but brief (<= 1 sentence). Do NOT answer the question.

Chat history (may include context):
{history}

Latest user question:
{question}

Refined query:
"""

def get_text_history_for_refiner(history) -> str:
    """Create a compact text view of previous turns for the refiner prompt."""
    parts = []
    for m in history:
        if m["role"] == "user":
            parts.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"AI: {m['content']}")
    return "\n".join(parts[-12:])  # last ~12 lines for brevity

def refine_query(history, user_question, model_name: str):
    """
    Returns (refined_text, off_topic_flag, error_message).
    Uses the same Groq model at temperature=0.0 for stable refinement.
    """
    try:
        refiner = ChatGroq(model_name=model_name, temperature=0.0)
        prompt = REFINER_PROMPT_TEMPLATE.format(
            history=get_text_history_for_refiner(history),
            question=user_question
        )
        out = refiner.invoke(prompt)
        refined = getattr(out, "content", "").strip() if out else ""
        if refined == "SORRY_OFF_TOPIC":
            return "", True, None
        # A little guardrail: if model returns something too long or empty, fall back
        if not refined or len(refined.split()) < 3 or len(refined) > 400:
            return user_question, False, None
        return refined, False, None
    except Exception as e:
        return user_question, False, f"{e}"

def ensure_state():
    if "chats" not in st.session_state:
        chat_id = str(uuid.uuid4())
        st.session_state.chats = {chat_id: create_empty_chat(new_chat_name(1))}
        st.session_state.active_chat_id = chat_id
        st.session_state.chat_counter = 1

ensure_state()

# ---------------- Sidebar: Chat Management ----------------
with st.sidebar:
    st.title("🩺 Healthcare Assistant (Groq)")
    st.caption("Answers **only** health-related questions.")

    # ✅ Valid public Groq models (no 404s)
    MODEL_CHOICES = [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ]
    model = st.selectbox(
        "Groq model",
        options=MODEL_CHOICES,
        index=0,
        help="Try: llama3-8b-8192, llama3-70b-8192, gemma2-9b-it, mixtral-8x7b-32768"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.markdown("### Refinement")
    use_refine = st.checkbox("Refine queries (beta)", value=True)
    show_refined = st.checkbox("Show refined query in chat", value=True)

    st.markdown("---")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.chat_counter += 1
        chat_id = str(uuid.uuid4())
        st.session_state.chats[chat_id] = create_empty_chat(new_chat_name(st.session_state.chat_counter))
        st.session_state.active_chat_id = chat_id
        st.rerun()

    chat_options = [(cid, data["name"]) for cid, data in st.session_state.chats.items()]
    ids = [cid for cid, _ in chat_options]
    names = [nm for _, nm in chat_options]
    current_idx = ids.index(st.session_state.active_chat_id) if st.session_state.active_chat_id in ids else 0

    selected_name = st.selectbox("Chats", names, index=current_idx)
    new_active_id = ids[names.index(selected_name)]
    if new_active_id != st.session_state.active_chat_id:
        st.session_state.active_chat_id = new_active_id
        st.rerun()

    current_chat = st.session_state.chats[st.session_state.active_chat_id]
    new_name = st.text_input("Rename current chat", value=current_chat["name"])
    if new_name.strip() and new_name != current_chat["name"]:
        current_chat["name"] = new_name

    cols = st.columns(2)
    if cols[0].button("🧹 Clear", use_container_width=True):
        current_chat["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()
    if cols[1].button("🗑️ Delete", use_container_width=True):
        if len(st.session_state.chats) > 1:
            del st.session_state.chats[st.session_state.active_chat_id]
            st.session_state.active_chat_id = next(iter(st.session_state.chats.keys()))
            st.rerun()
        else:
            st.warning("You need at least one chat.")

# ---------------- Groq LLM ----------------
llm = ChatGroq(model_name=model, temperature=temperature)

# ---------------- Main: Render History ----------------
active_chat = st.session_state.chats[st.session_state.active_chat_id]
for m in active_chat["messages"]:
    if m["role"] == "system":
        continue
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

# ---------------- Chat Input (Streaming) ----------------
user_input = st.chat_input("Ask a health-related question...")
if user_input:
    # 1) Show the original user message in UI and store it
    active_chat["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Optionally refine it
    refined_text = user_input
    off_topic = False
    refine_err = None
    if use_refine:
        refined_text, off_topic, refine_err = refine_query(active_chat["messages"], user_input, model)

    # 3) If off-topic, respond immediately with your exact apology and log it
    if off_topic:
        apology = 'Sorry, I can only answer healthcare-related questions.'
        with st.chat_message("assistant"):
            st.markdown(apology)
        active_chat["messages"].append({"role": "assistant", "content": apology})
    else:
        # 4) Build prompt; replace the *last* user message with refined_text for the model only
        prompt = build_prompt_from_history(active_chat["messages"], last_user_override=refined_text)

        with st.chat_message("assistant"):
            # Optionally show the refined query
            if use_refine and show_refined and refined_text != user_input:
                st.markdown(f"<div class='refined'>Refined query: <em>{refined_text}</em></div>", unsafe_allow_html=True)
            if use_refine and refine_err:
                st.markdown(f"<div class='refined'>Refiner note: {refine_err}</div>", unsafe_allow_html=True)

            # Stream the final answer
            placeholder = st.empty()
            ai_text = ""
            try:
                for chunk in llm.stream(prompt):
                    ai_text += getattr(chunk, "content", "") or ""
                    placeholder.markdown(ai_text)
            except Exception as e:
                ai_text = f"❌ Error: {e}"
                placeholder.error(ai_text)

        # 5) Save assistant message
        active_chat["messages"].append({"role": "assistant", "content": ai_text})
