import streamlit as st
import requests
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="PAKEmployee GPT",
    page_icon="🇵🇰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

API_URL = "http://localhost:8000/api/v1/query"

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* ── Header card ── */
.header-card {
    background: linear-gradient(135deg, #01411C 0%, #0a5c28 60%, #01411C 100%);
    border-radius: 16px;
    padding: 24px 32px;
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 4px 24px rgba(1,65,28,0.4);
}

.header-text h1 {
    color: #ffffff;
    font-size: 26px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.3px;
}

.header-text p {
    color: rgba(255,255,255,0.7);
    font-size: 13px;
    margin: 4px 0 0 0;
    letter-spacing: 0.5px;
}

.badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.9);
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 6px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Chat messages ── */
.user-msg {
    background: #f0f7f2;
    border: 1px solid #c8e6c9;
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-left: 15%;
    color: #1a3a22;
    font-size: 14px;
    line-height: 1.6;
}

.bot-msg {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-left: 3px solid #01411C;
    border-radius: 4px 14px 14px 14px;
    padding: 14px 16px;
    margin: 8px 0;
    margin-right: 10%;
    color: #1a1a1a;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.bot-msg strong { color: #01411C; }

.source-row {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #f0f0f0;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.source-chip {
    background: #e8f5e9;
    color: #2e7d32;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 10px;
    border: 1px solid #c8e6c9;
}

.meta-row {
    margin-top: 6px;
    font-size: 11px;
    color: #999;
}

.error-msg {
    background: #fff3f3;
    border: 1px solid #ffcdd2;
    border-left: 3px solid #e53935;
    border-radius: 4px 14px 14px 14px;
    padding: 14px 16px;
    margin: 8px 0;
    color: #c62828;
    font-size: 13px;
}

/* ── Role labels ── */
.role-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 3px;
    padding: 0 2px;
}

.role-user { color: #2e7d32; text-align: right; }
.role-bot  { color: #555; }

/* ── Input area ── */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #c8e6c9 !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #01411C !important;
    box-shadow: 0 0 0 2px rgba(1,65,28,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #01411C !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 8px 24px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #0a5c28 !important;
    transform: translateY(-1px) !important;
}

/* ── Suggestions ── */
.suggest-label {
    font-size: 12px;
    color: #888;
    margin-bottom: 8px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #f0f0f0;
    margin: 16px 0;
}

/* Status bar */
.status-bar {
    background: #f8fdf9;
    border: 1px solid #e0f0e3;
    border-radius: 10px;
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #555;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <div style="font-size: 52px; line-height:1;">🇵🇰</div>
    <div class="header-text">
        <h1>PAKEmployee GPT</h1>
        <p>Intelligent Legal Research · Employment & Contract Law</p>
        <span class="badge">⚖ Powered by Agentic RAG</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Check API health ──────────────────────────────────────────
def check_api():
    try:
        r = requests.get("http://localhost:8000/api/v1/health", timeout=3)
        return r.status_code == 200
    except:
        return False

api_ok = check_api()

if api_ok:
    st.markdown("""
    <div class="status-bar">
        <span style="color:#2e7d32; font-size:16px;">●</span>
        API connected &nbsp;·&nbsp; Documents indexed &nbsp;·&nbsp; Ready to answer
    </div>
    """, unsafe_allow_html=True)
else:
    st.error(
        "⚠️ Cannot reach API at localhost:8000. "
        "Run: `uvicorn api.main:app --host 0.0.0.0 --port 8000`",
        icon="🔴"
    )


# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Suggestion questions ──────────────────────────────────────
SUGGESTIONS = [
    "What is the notice period for terminating a permanent employee?",
    "What are the grounds for summary dismissal without notice?",
    "How is severance pay calculated under Pakistani labour law?",
    "What rights does an employee have during probation?",
]

if not st.session_state.messages:
    st.markdown('<div class="suggest-label">TRY ASKING</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)


# ── Render chat history ───────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="role-label role-user">You</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)

    elif msg["role"] == "assistant":
        st.markdown(f'<div class="role-label role-bot">PAKEmployee GPT</div>', unsafe_allow_html=True)

        answer_text = msg["content"]
        sources     = msg.get("sources", [])
        iterations  = msg.get("iterations", None)
        is_error    = msg.get("error", False)

        if is_error:
            st.markdown(f'<div class="error-msg">{answer_text}</div>', unsafe_allow_html=True)
        else:
            sources_html = ""
            if sources:
                chips = "".join(
                    f'<span class="source-chip">📄 {s["source"]} · p.{s["page"]}</span>'
                    for s in sources
                )
                sources_html = f'<div class="source-row">{chips}</div>'

            meta_html = ""
            if iterations:
                meta_html = f'<div class="meta-row">Agent used {iterations} search loop{"s" if iterations > 1 else ""}</div>'

            st.markdown(
                f'<div class="bot-msg">{answer_text}{sources_html}{meta_html}</div>',
                unsafe_allow_html=True
            )


# ── Input ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="question",
            placeholder="Ask about Pakistani employment or contract law...",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Send →", use_container_width=True)


# ── Handle pending suggestion click ──────────────────────────
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")
    submitted  = True


# ── Send to API ───────────────────────────────────────────────
def query_api(question: str) -> dict:
    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": True, "message": f"API error {response.status_code}: {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": True, "message": "Cannot connect to API. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"error": True, "message": "Request timed out. The agent may still be processing — try again."}
    except Exception as e:
        return {"error": True, "message": str(e)}


if submitted and user_input and user_input.strip():
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip()
    })

    # Query API with spinner
    with st.spinner("Searching Pakistani legal documents..."):
        result = query_api(user_input.strip())

    if result.get("error"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("message", "Something went wrong."),
            "error": True
        })
    else:
        # Strip inline source block from answer text (already shown as chips)
        answer = result.get("answer", "")
        clean_answer = answer.split("Sources:")[0].strip() if "Sources:" in answer else answer

        st.session_state.messages.append({
            "role": "assistant",
            "content": clean_answer,
            "sources": result.get("sources", []),
            "iterations": result.get("iterations"),
            "error": False
        })

    st.rerun()


# ── Footer ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; font-size:11px; color:#bbb;">'
    'PAKEmployee GPT · Agentic RAG · BGE Embeddings · Hybrid Retrieval · Built with ❤️ for Pakistan'
    '</div>',
    unsafe_allow_html=True
)