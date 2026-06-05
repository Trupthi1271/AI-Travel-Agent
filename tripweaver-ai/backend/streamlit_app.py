"""
TripWeaver AI — Streamlit Interface (Week 7-8 Polish)
======================================================
Production-ready UI with:
  - Clean chat interface with message history
  - Result display cards (weather, hotels, flights, budget)
  - Save / export functionality (JSON + TXT download)
  - Input validation with helpful error messages
  - Document upload + RAG Q&A
  - Saved trips panel
  - Responsive sidebar with tool status
"""

import os
import json
import uuid
import warnings
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TripWeaver AI ✈️",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "About": "TripWeaver AI — Your personal travel concierge for India.",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Card styling */
.result-card {
    background: #1E2130;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    border-left: 4px solid #FF6B35;
}
/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 0.5rem 0;
}
.metric-card {
    background: #262840;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
/* Input validation */
.validation-error {
    color: #FF4B4B;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}
/* Sidebar section headers */
.sidebar-header {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin: 0.5rem 0 0.3rem 0;
}
/* Tool badge */
.tool-badge {
    display: inline-block;
    background: #2D3250;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading AI agent…")
def get_planner():
    from agent.planner import TripPlanner
    return TripPlanner()


@st.cache_resource(show_spinner=False)
def get_doc_store():
    from rag.document_store import TravelDocumentStore
    return TravelDocumentStore()


# ── Session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "doc_names": [],
        "_quick_query": None,
        "session_id": str(uuid.uuid4())[:8],
        "saved_trips": [],          # list of {title, content, timestamp}
        "active_tab": "chat",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Input validation ──────────────────────────────────────────────────────────
def validate_input(text: str) -> tuple[bool, str]:
    """
    Validate user input before sending to agent.
    Returns (is_valid, error_message).
    """
    text = text.strip()
    if not text:
        return False, "Please enter a message."
    if len(text) < 3:
        return False, "Message too short — please be more specific."
    if len(text) > 1000:
        return False, f"Message too long ({len(text)} chars). Please keep it under 1000 characters."
    # Block obvious non-travel queries
    blocked = ["password", "credit card", "ssn", "social security", "bank account"]
    if any(b in text.lower() for b in blocked):
        return False, "Please only ask travel-related questions."
    return True, ""


# ── Response routing ──────────────────────────────────────────────────────────
def _get_response(user_input: str) -> tuple[str, dict]:
    """Returns (response_text, trace_dict)."""
    import time
    from agent.graph import run_graph
    from langchain_core.messages import HumanMessage, AIMessage

    start = time.perf_counter()
    doc_store = get_doc_store()
    session_id = st.session_state.session_id

    # RAG path
    if doc_store.has_documents():
        agent_keywords = [
            "weather", "hotel", "budget", "plan a trip", "calculate",
            "forecast", "temperature", "accommodation", "flight", "fly",
        ]
        if not any(kw in user_input.lower() for kw in agent_keywords):
            from rag.rag_chain import answer_from_docs
            rag_response = answer_from_docs(user_input, doc_store)
            if rag_response:
                latency = round((time.perf_counter() - start) * 1000)
                return (
                    "## 📄 From Your Documents\n\n" + rag_response,
                    {"path": "RAG", "latency_ms": latency, "docs": doc_store.document_names}
                )

    # Convert history to LangChain messages
    chat_history = []
    for m in st.session_state.messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            chat_history.append(HumanMessage(content=content))
        elif role == "assistant":
            chat_history.append(AIMessage(content=content))

    # Run LangGraph workflow
    response, graph_trace = run_graph(user_input, chat_history, session_id=session_id)
    latency = round((time.perf_counter() - start) * 1000)

    # Merge graph trace with metrics
    trace = {
        **graph_trace,
        "latency_ms": latency,   # use end-to-end latency from Streamlit side
    }
    return response, trace


# ── Export helpers ────────────────────────────────────────────────────────────
def _export_chat_txt() -> str:
    """Format full conversation as plain text."""
    lines = [
        "TripWeaver AI — Conversation Export",
        f"Session: {st.session_state.session_id}",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
        "",
    ]
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "TripWeaver"
        lines.append(f"[{role}]")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def _export_chat_json() -> str:
    """Format conversation as JSON."""
    return json.dumps({
        "session_id": st.session_state.session_id,
        "exported_at": datetime.now().isoformat(),
        "messages": st.session_state.messages,
        "saved_trips": st.session_state.saved_trips,
    }, indent=2, ensure_ascii=False)


def _save_last_response():
    """Save the last assistant response as a trip card."""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            # Extract a title from the first heading or first line
            content = msg["content"]
            title = "Trip Plan"
            for line in content.split("\n"):
                line = line.strip().lstrip("#").strip()
                if line and len(line) < 60:
                    title = line
                    break
            st.session_state.saved_trips.append({
                "title": title,
                "content": content,
                "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
            })
            return True
    return False


# ── Detect response type for card rendering ───────────────────────────────────
def _detect_response_type(content: str) -> str:
    c = content.lower()
    # Check for specific section headers first (most reliable)
    if "## 🌤️ weather" in c or "## 🌤 weather" in c:
        return "weather"
    if "## 🏨 hotels" in c:
        return "hotel"
    if "## ✈️ flights" in c:
        return "flight"
    if "## 💰 budget breakdown" in c:
        return "budget"
    if "## 🗺️" in c and ("day 1" in c or "day-by-day" in c or "trip to" in c):
        return "itinerary"
    if "## 🗺️ top places" in c:
        return "places"
    # Fallback: content-based detection
    if "°c" in c and ("humidity" in c or "forecast" in c):
        return "weather"
    if "economy" in c and "→" in c and "₹" in c:
        return "flight"
    if "accommodation" in c and "food" in c and "transport" in c and "₹" in c:
        return "budget"
    if "day 1" in c and "morning" in c:
        return "itinerary"
    return "general"


def _render_response_card(content: str):
    """Render assistant response with a styled card header based on type."""
    rtype = _detect_response_type(content)
    icons = {
        "weather":   ("🌤️", "#4A90D9", "Weather Report"),
        "hotel":     ("🏨", "#27AE60", "Accommodation Options"),
        "flight":    ("✈️", "#8E44AD", "Flight Results"),
        "budget":    ("💰", "#E67E22", "Budget Breakdown"),
        "itinerary": ("🗺️", "#FF6B35", "Trip Itinerary"),
        "general":   ("💬", "#555", ""),
    }
    icon, color, label = icons.get(rtype, icons["general"])

    if label:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'margin-bottom:8px;padding:6px 12px;background:{color}22;'
            f'border-radius:6px;border-left:3px solid {color}">'
            f'<span style="font-size:1.2rem">{icon}</span>'
            f'<span style="font-weight:600;color:{color}">{label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown(content)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌍 TripWeaver AI")
    st.caption("Your AI Travel Concierge for India")
    st.divider()

    # ── Navigation tabs ───────────────────────────────────────────────────────
    tab_chat, tab_saved, tab_docs = st.tabs(["💬 Chat", "💾 Saved", "📄 Docs"])

    # ── Chat tab ──────────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown('<p class="sidebar-header">⚡ Quick Actions</p>', unsafe_allow_html=True)
        quick_actions = {
            "🌤 Weather in Goa":        "What's the weather in Goa?",
            "🏨 Hotels in Jaipur":      "Suggest hotels in Jaipur",
            "✈️ Flights Delhi → Goa":   "Flights from Delhi to Goa",
            "💰 Budget ₹15k / 3 days":  "My budget is 15000 INR for 3 days",
            "📅 Plan Manali 3 days":     "Plan a 3-day budget trip to Manali",
            "🗺️ Places in Varanasi":    "Top places to visit in Varanasi",
        }
        cols = st.columns(2)
        for i, (label, query) in enumerate(quick_actions.items()):
            if cols[i % 2].button(label, use_container_width=True, key=f"qa_{i}"):
                st.session_state._quick_query = query

        st.divider()

        # Export buttons
        st.markdown('<p class="sidebar-header">📥 Export</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.messages:
                st.download_button(
                    "📝 TXT",
                    data=_export_chat_txt(),
                    file_name=f"tripweaver_{st.session_state.session_id}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        with col2:
            if st.session_state.messages:
                st.download_button(
                    "📊 JSON",
                    data=_export_chat_json(),
                    file_name=f"tripweaver_{st.session_state.session_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            get_planner().clear_memory()
            st.rerun()

    # ── Saved trips tab ───────────────────────────────────────────────────────
    with tab_saved:
        if not st.session_state.saved_trips:
            st.info("No saved trips yet.\n\nAfter getting a response, click **💾 Save** to bookmark it here.")
        else:
            for i, trip in enumerate(reversed(st.session_state.saved_trips)):
                with st.expander(f"📌 {trip['title'][:35]}", expanded=False):
                    st.caption(f"Saved: {trip['timestamp']}")
                    st.markdown(trip["content"])
                    st.download_button(
                        "⬇️ Download",
                        data=trip["content"],
                        file_name=f"trip_{i+1}.txt",
                        mime="text/plain",
                        key=f"dl_trip_{i}",
                        use_container_width=True,
                    )

            if st.button("🗑️ Clear Saved Trips", use_container_width=True):
                st.session_state.saved_trips = []
                st.rerun()

    # ── Docs tab ──────────────────────────────────────────────────────────────
    with tab_docs:
        st.caption("Upload PDFs or TXT files to ask questions about them.")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            doc_store = get_doc_store()
            for uf in uploaded_files:
                if uf.name not in st.session_state.doc_names:
                    with st.spinner(f"Indexing {uf.name}…"):
                        suffix = Path(uf.name).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uf.read())
                            tmp_path = tmp.name
                        chunks = doc_store.add_file(tmp_path, uf.name)
                        os.unlink(tmp_path)
                    st.session_state.doc_names.append(uf.name)
                    st.success(f"✅ {uf.name} — {chunks} chunks")

        if st.session_state.doc_names:
            st.divider()
            st.caption("**Indexed:**")
            for name in st.session_state.doc_names:
                st.markdown(f"📎 `{name}`")
            if st.button("🗑️ Clear Documents", use_container_width=True):
                get_doc_store().clear()
                st.session_state.doc_names = []
                st.rerun()

    # ── Tool status ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="sidebar-header">🛠️ Active Tools</p>', unsafe_allow_html=True)
    from agent.tools import TOOL_METADATA
    for meta in TOOL_METADATA.values():
        st.markdown(
            f'{meta["icon"]} **{meta["label"]}** '
            f'<span class="tool-badge">{meta["api"]}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    # ── Monitoring panel ──────────────────────────────────────────────────
    st.markdown('<p class="sidebar-header">📊 Monitoring</p>', unsafe_allow_html=True)
    try:
        from agent.logger import metrics
        summary = metrics.summary()
        col_a, col_b = st.columns(2)
        col_a.metric("Runs", summary["agent_runs"])
        col_b.metric("Errors", summary["agent_errors"])
        if summary["avg_agent_ms"]:
            st.caption(f"Avg response: {summary['avg_agent_ms']}ms")
        if summary["tool_calls"]:
            top_tool = max(summary["tool_calls"], key=summary["tool_calls"].get)
            st.caption(f"Most used: {top_tool} ({summary['tool_calls'][top_tool]}x)")
    except Exception:
        pass

    st.divider()
    st.caption(
        f"Session `{st.session_state.session_id}` · "
        "Powered by Groq · LangChain · LangGraph · Amadeus · Weatherstack"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_stats = st.columns([3, 1])
with col_title:
    st.markdown("# ✈️ TripWeaver AI")
    st.caption("Your intelligent travel concierge for India — powered by Groq & LangChain")
with col_stats:
    msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    saved_count = len(st.session_state.saved_trips)
    st.metric("Queries", msg_count)
    st.metric("Saved Trips", saved_count)

st.divider()

# ── Welcome banner (shown only on first load) ─────────────────────────────────
if not st.session_state.messages:
    with st.container():
        st.markdown("""
### 👋 Welcome! Here's what I can do:

| | Capability | Try asking… |
|---|---|---|
| 🌤️ | **Live Weather** | *What's the weather in Manali?* |
| 🏨 | **Hotel Finder** | *Suggest hotels in Jaipur* |
| ✈️ | **Flight Search** | *Flights from Delhi to Goa* |
| 💰 | **Budget Planner** | *My budget is ₹20,000 for 4 days* |
| 🗺️ | **Trip Itinerary** | *Plan a 3-day trip to Kerala* |
| 🔍 | **Travel Search** | *Best time to visit Ladakh?* |
| 📄 | **Doc Q&A** | *Upload a PDF and ask questions* |

Use the **Quick Actions** in the sidebar or type below to get started.
        """)
        st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🌍"):
        if msg["role"] == "assistant":
            _render_response_card(msg["content"])
            # Trace expander — shows what the agent did
            if msg.get("trace"):
                with st.expander("🔍 How this was answered", expanded=False):
                    st.json(msg["trace"])
            # Save button below each assistant message
            col_save, col_empty = st.columns([1, 4])
            with col_save:
                if st.button("💾 Save", key=f"save_{i}", help="Save this response to your trips"):
                    # Temporarily set messages to just up to this point for saving
                    saved_content = msg["content"]
                    title = "Trip Plan"
                    for line in saved_content.split("\n"):
                        line = line.strip().lstrip("#").strip()
                        if line and len(line) < 60:
                            title = line
                            break
                    st.session_state.saved_trips.append({
                        "title": title,
                        "content": saved_content,
                        "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
                    })
                    st.toast("✅ Saved to your trips!", icon="💾")
        else:
            st.markdown(msg["content"])

# ── Input area ────────────────────────────────────────────────────────────────
user_input: str | None = None

# Handle quick action injection
if st.session_state._quick_query:
    user_input = st.session_state._quick_query
    st.session_state._quick_query = None

# Chat input with placeholder
chat_input = st.chat_input(
    "Ask me about your trip — weather, hotels, flights, budget, itinerary…",
)
if chat_input:
    user_input = chat_input

# ── Process input ─────────────────────────────────────────────────────────────
if user_input:
    # Validate input
    is_valid, error_msg = validate_input(user_input)
    if not is_valid:
        st.error(f"⚠️ {error_msg}")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # Generate and show response
        with st.chat_message("assistant", avatar="🌍"):
            with st.spinner("🤔 Thinking…"):
                response, trace = _get_response(user_input)
            _render_response_card(response)

            # Query trace expander
            with st.expander("🔍 How this was answered", expanded=False):
                import json
                st.code(json.dumps(trace, indent=2, default=str), language="json")

            # Save button for new response
            col_save, _ = st.columns([1, 4])
            with col_save:
                if st.button("💾 Save", key="save_new", help="Save this response"):
                    title = "Trip Plan"
                    for line in response.split("\n"):
                        line = line.strip().lstrip("#").strip()
                        if line and len(line) < 60:
                            title = line
                            break
                    st.session_state.saved_trips.append({
                        "title": title,
                        "content": response,
                        "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
                    })
                    st.toast("✅ Saved to your trips!", icon="💾")

        st.session_state.messages.append({"role": "assistant", "content": response, "trace": trace})
        st.rerun()
