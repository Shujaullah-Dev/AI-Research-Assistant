import uuid
import time
from datetime import datetime

import requests
import streamlit as st


# -------------------------------------------------------------------
# Application configuration
# -------------------------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000"

UPLOAD_ENDPOINT = f"{API_BASE_URL}/documents/upload"
ASK_ENDPOINT = f"{API_BASE_URL}/ask"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

MAX_UPLOAD_SIZE = 20 * 1024 * 1024
MAX_QUESTION_LENGTH = 1000


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Marginalia — Research Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Design system
#
# A modern reading interface for research papers: a deep indigo accent
# on a clean paper background, soft layered shadows, rounded surfaces,
# and a serif voice reserved for anything meant to be read (masthead,
# questions, answers) while a plain grotesque carries the controls.
# -------------------------------------------------------------------

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400;1,8..60,500&display=swap');

    :root {
        --ink: #171B26;
        --ink-soft: #525B6E;
        --ink-faint: #8B93A5;
        --paper: #F3F4F9;
        --surface: #FFFFFF;
        --surface-2: #F7F8FC;
        --line: #E2E5EE;
        --line-strong: #C6CCDC;
        --accent: #4C5FD5;
        --accent-strong: #3A49B4;
        --accent-soft: rgba(76, 95, 213, 0.09);
        --accent-border: rgba(76, 95, 213, 0.38);
        --teal: #0E8A72;
        --teal-soft: rgba(14, 138, 114, 0.10);
        --brick: #B4452F;
        --brick-soft: rgba(180, 69, 47, 0.10);
        --radius: 14px;
        --radius-sm: 9px;
        --shadow-sm: 0 1px 2px rgba(23, 27, 38, 0.05);
        --shadow-md: 0 6px 16px rgba(23, 27, 38, 0.07);
        --shadow-lg: 0 16px 40px rgba(23, 27, 38, 0.12);
        --font-sans: 'Inter', -apple-system, 'Segoe UI', sans-serif;
        --font-serif: 'Source Serif 4', Georgia, serif;
    }

    html, body, [class*="css"] {
        font-family: var(--font-sans);
        color: var(--ink);
    }

    .stApp {
        background: var(--paper);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1120px;
    }

    ::selection {
        background: var(--accent-soft);
    }

    /* -------------------------------------------------------------
       Masthead
       ------------------------------------------------------------- */

    @keyframes settle {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .masthead {
        position: relative;
        overflow: hidden;
        border-radius: 18px;
        padding: 2.4rem 2.6rem 2.1rem 2.6rem;
        margin-bottom: 2rem;
        background: linear-gradient(125deg, #1C2242 0%, #2A3160 48%, #3A49B4 100%);
        color: #FFFFFF;
        box-shadow: var(--shadow-lg);
        animation: settle 0.55s ease-out;
    }

    .masthead::after {
        content: "";
        position: absolute;
        top: -40%;
        right: -12%;
        width: 46%;
        height: 180%;
        background: radial-gradient(
            closest-side,
            rgba(124, 140, 240, 0.35),
            transparent
        );
        pointer-events: none;
    }

    .masthead-kicker {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #C7CFFF;
        margin-bottom: 0.7rem;
    }

    .masthead-title {
        font-family: var(--font-serif);
        font-size: 2.9rem;
        font-weight: 500;
        line-height: 1.05;
        letter-spacing: -0.015em;
        margin-bottom: 0.55rem;
    }

    .masthead-subtitle {
        font-family: var(--font-serif);
        font-style: italic;
        font-size: 1.18rem;
        line-height: 1.5;
        color: rgba(255, 255, 255, 0.80);
        max-width: 52ch;
        margin-bottom: 1.3rem;
    }

    .masthead-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }

    .masthead-badge {
        font-size: 0.74rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        color: rgba(255, 255, 255, 0.92);
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        padding: 0.28rem 0.75rem;
        backdrop-filter: blur(4px);
    }

    /* -------------------------------------------------------------
       Section headers
       ------------------------------------------------------------- */

    .section-title {
        font-family: var(--font-serif);
        font-size: 1.55rem;
        font-weight: 500;
        color: var(--ink);
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }

    .section-title::before {
        content: "";
        width: 4px;
        height: 1.25rem;
        border-radius: 4px;
        background: var(--accent);
        flex-shrink: 0;
    }

    .section-description {
        color: var(--ink-soft);
        font-size: 0.94rem;
        line-height: 1.55;
        margin-bottom: 1.15rem;
        max-width: 64ch;
    }

    /* -------------------------------------------------------------
       Document statistics cards
       ------------------------------------------------------------- */

    .stat-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 3px solid var(--accent);
        border-radius: var(--radius);
        padding: 1rem 1.15rem;
        min-height: 96px;
        box-shadow: var(--shadow-sm);
        transition: transform 0.16s ease, box-shadow 0.16s ease;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--ink-faint);
        margin-bottom: 0.4rem;
    }

    .stat-value {
        font-family: var(--font-serif);
        font-size: 1.35rem;
        font-weight: 500;
        line-height: 1.25;
        word-break: break-word;
        color: var(--ink);
    }

    /* -------------------------------------------------------------
       Answer container
       ------------------------------------------------------------- */

    .answer-container {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: var(--radius);
        padding: 1.5rem 1.7rem;
        margin-top: 0.6rem;
        margin-bottom: 1.1rem;
        box-shadow: var(--shadow-md);
    }

    .answer-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent-strong);
        margin-bottom: 0.55rem;
    }

    .answer-timestamp {
        font-size: 0.78rem;
        color: var(--ink-faint);
        margin-bottom: 0.85rem;
    }

    .answer-body {
        font-family: var(--font-serif);
        font-size: 1.13rem;
        line-height: 1.7;
        color: var(--ink);
        white-space: pre-wrap;
    }

    /* -------------------------------------------------------------
       History question header
       ------------------------------------------------------------- */

    .history-question {
        font-family: var(--font-serif);
        font-style: italic;
        font-size: 1.08rem;
        color: var(--ink);
        border-left: 3px solid var(--accent-border);
        padding-left: 0.95rem;
        margin-bottom: 0.6rem;
        padding-top: 0.7rem;
    }

    /* -------------------------------------------------------------
       Citation cards
       ------------------------------------------------------------- */

    .citation-card {
        background: var(--surface-2);
        border: 1px solid var(--line);
        border-radius: var(--radius-sm);
        padding: 0.85rem 1.05rem;
        margin-bottom: 0.65rem;
        box-shadow: var(--shadow-sm);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .citation-card:hover {
        border-color: var(--accent-border);
        box-shadow: var(--shadow-md);
    }

    .citation-title {
        font-family: var(--font-serif);
        font-size: 1.04rem;
        font-weight: 500;
        color: var(--ink);
        margin-bottom: 0.45rem;
    }

    .citation-number {
        color: var(--accent);
        font-style: italic;
        margin-right: 0.2rem;
    }

    .citation-meta {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .chip {
        font-size: 0.76rem;
        font-weight: 500;
        color: var(--accent-strong);
        background: var(--accent-soft);
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        white-space: nowrap;
    }

    /* -------------------------------------------------------------
       Sidebar
       ------------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, var(--surface-2) 100%);
        border-right: 1px solid var(--line);
    }

    .sidebar-brand {
        font-family: var(--font-serif);
        font-size: 1.55rem;
        font-weight: 500;
        color: var(--ink);
        margin-bottom: 0.15rem;
    }

    .sidebar-brand-rule {
        width: 2.6rem;
        height: 3px;
        border-radius: 3px;
        background: var(--accent);
        margin-bottom: 0.75rem;
    }

    .sidebar-description {
        font-size: 0.88rem;
        line-height: 1.6;
        color: var(--ink-soft);
    }

    .sidebar-heading {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--ink-faint);
        margin-bottom: 0.45rem;
    }

    .sidebar-step {
        display: flex;
        align-items: baseline;
        gap: 0.55rem;
        font-size: 0.88rem;
        color: var(--ink);
        margin-bottom: 0.5rem;
        line-height: 1.45;
    }

    .sidebar-step-number {
        flex-shrink: 0;
        width: 1.35rem;
        height: 1.35rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--accent-strong);
        background: var(--accent-soft);
        border-radius: 50%;
    }

    .sidebar-item {
        font-size: 0.86rem;
        color: var(--ink-soft);
        line-height: 1.6;
    }

    .sidebar-item strong {
        color: var(--accent-strong);
        font-family: var(--font-serif);
        font-style: italic;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.35; }
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .status-online {
        background: var(--teal-soft);
        color: var(--teal);
    }

    .status-offline {
        background: var(--brick-soft);
        color: var(--brick);
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse 2.2s ease-in-out infinite;
    }

    /* -------------------------------------------------------------
       Footer
       ------------------------------------------------------------- */

    .app-footer {
        text-align: left;
        padding-top: 1.4rem;
        margin-top: 3rem;
        border-top: 1px solid var(--line);
        color: var(--ink-faint);
        font-size: 0.84rem;
    }

    /* -------------------------------------------------------------
       Streamlit widget overrides
       ------------------------------------------------------------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.8rem;
        border: 1px solid var(--ink);
        background: var(--ink);
        color: var(--paper);
        transition: background 0.15s ease, color 0.15s ease,
                    border-color 0.15s ease, transform 0.1s ease;
    }

    .stButton > button:hover {
        background: var(--accent);
        border-color: var(--accent);
        color: #FFFFFF;
    }

    .stButton > button:active {
        transform: translateY(1px);
    }

    .stButton > button:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }

    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid var(--line-strong);
        background: var(--surface);
        color: var(--ink);
        font-weight: 600;
        min-height: 2.6rem;
        transition: border-color 0.15s ease, color 0.15s ease;
    }

    .stDownloadButton > button:hover {
        border-color: var(--accent);
        color: var(--accent-strong);
    }

    [data-testid="stFileUploader"] {
        border-radius: var(--radius);
    }

    [data-testid="stFileDropZone"] {
        border: 2px dashed var(--line-strong);
        border-radius: var(--radius);
        background: var(--surface-2);
        transition: border-color 0.15s ease, background 0.15s ease;
    }

    [data-testid="stFileDropZone"]:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
    }

    .stTextArea textarea {
        font-family: var(--font-serif);
        font-size: 1.04rem;
        border-radius: 10px;
        border: 1px solid var(--line);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px var(--accent-soft);
    }

    .stTextArea label {
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.25rem;
        border-bottom: 2px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-serif);
        font-size: 1.02rem;
        color: var(--ink-soft);
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 0.25rem;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent-strong) !important;
        font-weight: 600;
    }

    details[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: var(--radius-sm);
        background: var(--surface);
    }

    details[data-testid="stExpander"] summary {
        font-weight: 600;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent), #7C8CF0);
        border-radius: 999px;
    }

    hr {
        border-color: var(--line) !important;
    }

    .stCaption, caption {
        color: var(--ink-faint);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Session state initialization
# -------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "uploaded_document" not in st.session_state:
    st.session_state["uploaded_document"] = None

if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = None

if "last_citations" not in st.session_state:
    st.session_state["last_citations"] = []

if "qa_history" not in st.session_state:
    # List of dicts: {question, answer, citations, timestamp}
    st.session_state["qa_history"] = []

if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""

def get_session_id():

    """
    Return the unique identifier for the current
    Streamlit session.
    """
    return st.session_state["session_id"]

session_id = get_session_id()

# -------------------------------------------------------------------
# API helper functions
# -------------------------------------------------------------------

def upload_document(uploaded_file):
    """
    Send the uploaded PDF to the FastAPI backend.
    """

    file_bytes = uploaded_file.getvalue()

    files = {
        "file": (
            uploaded_file.name,
            file_bytes,
            "application/pdf",
        )
    }
    data = {
        "session_id": get_session_id(),
    }

    response = requests.post(
        UPLOAD_ENDPOINT,
        files=files,
        data=data,
        timeout=300,
    )

    return response


def ask_question(question):
    """
    Send a question to the FastAPI backend.
    """

    payload = {
        "question": question,
        "session_id": get_session_id(),
        "top_k": 5,
        "final_k": 5,
    }

    response = requests.post(
        ASK_ENDPOINT,
        json=payload,
        timeout=300,
    )

    return response


def check_backend_health():
    """
    Ping the backend to check whether it is reachable.
    Returns True/False. Falls back to hitting the base URL
    if a dedicated /health endpoint doesn't exist.
    """

    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=3)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass

    try:
        response = requests.get(API_BASE_URL, timeout=3)
        return response.status_code < 500
    except requests.exceptions.RequestException:
        return False


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

def render_sidebar():
    """
    Render application information in the sidebar.
    """

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-brand">Marginalia</div>
            <div class="sidebar-brand-rule"></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-description">
                A reading companion for research papers —
                ask a question, get an answer grounded in
                the text, with the passage it came from.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------------------
        # Backend status
        # ---------------------------------------------------------

        st.markdown(
            '<div class="sidebar-heading">Backend</div>',
            unsafe_allow_html=True,
        )

        backend_online = check_backend_health()

        if backend_online:
            st.markdown(
                '<span class="status-pill status-online">'
                '<span class="status-dot"></span> Connected</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill status-offline">'
                '<span class="status-dot"></span> Unreachable</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Expected at {API_BASE_URL}")

        if st.button("Refresh status", use_container_width=True):
            st.rerun()

        st.divider()

        st.markdown(
            '<div class="sidebar-heading">How it works</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-step">
                <span class="sidebar-step-number">1</span>
                <span>Upload a research paper</span>
            </div>
            <div class="sidebar-step">
                <span class="sidebar-step-number">2</span>
                <span>Extract and index the content</span>
            </div>
            <div class="sidebar-step">
                <span class="sidebar-step-number">3</span>
                <span>Ask a question</span>
            </div>
            <div class="sidebar-step">
                <span class="sidebar-step-number">4</span>
                <span>Retrieve relevant passages</span>
            </div>
            <div class="sidebar-step">
                <span class="sidebar-step-number">5</span>
                <span>Generate a grounded answer</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            '<div class="sidebar-heading">System</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-item">
                <strong>FastAPI</strong> backend<br>
                <strong>BGE</strong> embeddings<br>
                <strong>FAISS</strong> retrieval<br>
                <strong>Cross-encoder</strong> reranking<br>
                <strong>Ollama</strong> LLM
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------------------
        # Session controls
        # ---------------------------------------------------------

        if st.session_state["uploaded_document"] is not None:

            st.divider()

            st.markdown(
                '<div class="sidebar-heading">Session</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                f"{len(st.session_state['qa_history'])} question(s) asked"
            )

            if st.button(
                "Start new session",
                use_container_width=True,
                help="Clear the current document and question history",
            ):
                st.session_state["session_id"] = str(uuid.uuid4())
                st.session_state["uploaded_document"] = None
                st.session_state["last_answer"] = None
                st.session_state["last_citations"] = []
                st.session_state["qa_history"] = []
                st.session_state["question_input"] = ""
                st.rerun()

        st.divider()

        st.caption("Local retrieval-augmented generation")


# -------------------------------------------------------------------
# Document information
# -------------------------------------------------------------------

def display_upload_result(upload_response):
    """
    Display information returned by the upload endpoint.
    """

    document_name = upload_response.get(
        "document_name",
        "Unknown",
    )

    pages_extracted = upload_response.get(
        "pages_extracted",
        "N/A",
    )

    chunks_indexed = upload_response.get(
        "chunks_indexed",
        "N/A",
    )

    embedding_model = upload_response.get(
        "embedding_model",
        "N/A",
    )

    vector_store = upload_response.get(
        "vector_store",
        "N/A",
    )

    status = upload_response.get(
        "status",
        "Unknown",
    )

    st.success(
        "Research paper processed successfully."
    )

    st.markdown(
        '<div class="section-title">Document overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Extracted, chunked, embedded, and added to the
            retrieval index.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Document</div>
                <div class="stat-value">{document_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Pages</div>
                <div class="stat-value">{pages_extracted}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Chunks</div>
                <div class="stat-value">{chunks_indexed}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Status</div>
                <div class="stat-value">{status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Technical document details"):

        st.write(
            f"**Embedding model:** {embedding_model}"
        )

        st.write(
            f"**Vector store:** {vector_store}"
        )


# -------------------------------------------------------------------
# Citation display
# -------------------------------------------------------------------

def display_citations(citations, key_prefix=""):
    """
    Display grounded citations returned by the API.
    """

    st.markdown(
        '<div class="section-title">Sources</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Passages retrieved from the indexed document
            that support the answer above.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not citations:
        st.info(
            "No citations were returned for this answer."
        )

        return

    for citation_number, citation in enumerate(
        citations,
        start=1,
    ):
        document_name = citation.get(
            "document_name",
            "Unknown document",
        )

        page_number = citation.get(
            "page_number",
            "N/A",
        )

        chunk_id = citation.get(
            "chunk_id",
            "N/A",
        )

        score = citation.get(
            "score",
            None,
        )

        score_text = (
            f"Relevance {score:.2f}"
            if score is not None
            else "Relevance unavailable"
        )

        st.markdown(
            f"""
            <div class="citation-card">
                <div class="citation-title">
                    <span class="citation-number">{citation_number:02d}</span>
                    {document_name}
                </div>
                <div class="citation-meta">
                    <span class="chip">Page {page_number}</span>
                    <span class="chip">Chunk {chunk_id}</span>
                    <span class="chip">{score_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Optional: show the retrieved passage text if the API returns it.
        chunk_text = citation.get("text") or citation.get("chunk_text")

        if chunk_text:
            with st.expander(
                f"Read passage {citation_number:02d}",
                expanded=False,
            ):
                st.markdown(chunk_text)


# -------------------------------------------------------------------
# Answer display
# -------------------------------------------------------------------

def display_answer(answer, citations, timestamp=None, key_prefix=""):
    """
    Display the generated answer and its citations.
    """

    if not answer or not answer.strip():

        st.warning(
            "The system did not return an answer."
        )

        return

    timestamp_html = (
        f'<div class="answer-timestamp">Answered at {timestamp}</div>'
        if timestamp
        else ""
    )

    st.markdown(
        f"""
        <div class="answer-container">
            <div class="answer-label">Grounded response</div>
            {timestamp_html}
            <div class="answer-body">{answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 5])

    with col_a:
        st.download_button(
            "Save answer",
            data=answer,
            file_name="answer.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"{key_prefix}_download",
        )

    display_citations(citations, key_prefix=key_prefix)


# -------------------------------------------------------------------
# Upload section
# -------------------------------------------------------------------

def render_upload_section():
    """
    Render the PDF upload and processing interface.
    """

    st.markdown(
        '<div class="section-title">Upload a research paper</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Upload a PDF to prepare it for question answering.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF research paper",
        type=["pdf"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:

        st.info(
            "Upload a PDF to begin your research session."
        )

        return

    file_name = uploaded_file.name
    file_size = len(
        uploaded_file.getvalue()
    )

    if not file_name.lower().endswith(".pdf"):

        st.error(
            "Invalid file type. Please upload a PDF file."
        )

        return

    if file_size == 0:

        st.error(
            "The uploaded file is empty."
        )

        return

    if file_size > MAX_UPLOAD_SIZE:

        st.error(
            "The file is too large. "
            "The maximum allowed size is 20 MB."
        )

        return

    file_size_mb = file_size / (1024 * 1024)
    st.caption(
        f"Selected: {file_name} · "
        f"{file_size_mb:.2f} MB"
    )

    upload_button = st.button(
        "Upload and process PDF",
        type="primary",
        use_container_width=True,
    )

    if upload_button:

        progress_bar = st.progress(
            0, text="Uploading document..."
        )

        try:

            progress_bar.progress(25, text="Uploading document...")

            response = upload_document(
                uploaded_file
            )

            progress_bar.progress(
                75, text="Indexing document..."
            )

            if response.status_code == 200:

                upload_response = response.json()

                st.session_state[
                    "uploaded_document"
                ] = upload_response

                st.session_state[
                    "last_answer"
                ] = None

                st.session_state[
                    "last_citations"
                ] = []

                st.session_state["qa_history"] = []
                progress_bar.progress(100, text="Done.")
                time.sleep(0.3)

                st.rerun()

            else:

                progress_bar.empty()

                try:
                    error_data = response.json()
                    error_message = error_data.get(
                        "detail",
                        "The upload request failed.",
                    )

                except ValueError:
                    error_message = (
                        "The server returned "
                        "an invalid error response."
                    )

                st.error(
                    f"Upload failed: {error_message}"
                )

        except requests.exceptions.ConnectionError:
            progress_bar.empty()
            st.error(
                "Could not connect to the FastAPI backend. "
                "Please confirm that the API server is running."
            )

        except requests.exceptions.Timeout:
            progress_bar.empty()
            st.error(
                "The request timed out while processing "
                "the PDF. Please try again."
            )

        except requests.exceptions.RequestException as error:
            progress_bar.empty()
            st.error(
                f"An unexpected network error occurred: {error}"
            )

        except Exception as error:
            progress_bar.empty()
            st.error(
                f"An unexpected error occurred: {error}"
            )

    if st.session_state["uploaded_document"] is not None:
        display_upload_result(
            st.session_state["uploaded_document"]
        )


# -------------------------------------------------------------------
# Question section
# -------------------------------------------------------------------

def render_question_section():
    """
    Render the question and answer interface.
    """

    st.markdown(
        '<div class="section-title">Ask about your paper</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            The assistant retrieves the relevant passages first,
            then writes an answer grounded in what it found.
        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Question",
        placeholder=(
            "What dataset did the researchers use to train the model?"
        ),
        height=110,
        max_chars=MAX_QUESTION_LENGTH,
        label_visibility="collapsed",
        key="question_input",
    )

    st.caption(
        f"{len(question)}/{MAX_QUESTION_LENGTH} characters"
    )

    col_ask, col_clear = st.columns([4, 1])

    with col_ask:
        ask_button = st.button(
            "Ask question",
            type="primary",
            use_container_width=True,
        )

    with col_clear:
        clear_button = st.button(
            "Clear",
            use_container_width=True,
        )

    if clear_button:
        st.session_state["question_input"] = ""
        st.rerun()

    if ask_button:

        if not question.strip():
            st.warning(
                "Please enter a question before submitting."
            )

            return

        with st.spinner(
            "Searching the paper and generating an answer..."
        ):

            try:

                response = ask_question(
                    question.strip()
                )

                if response.status_code == 200:

                    answer_response = response.json()
                    answer = answer_response.get(
                        "answer",
                        "",
                    )

                    citations = answer_response.get(
                        "citations",
                        [],
                    )

                    st.session_state[
                        "last_answer"
                    ] = answer

                    st.session_state[
                        "last_citations"
                    ] = citations

                    st.session_state["qa_history"].insert(
                        0,
                        {
                            "question": question.strip(),
                            "answer": answer,
                            "citations": citations,
                            "timestamp": datetime.now().strftime(
                                "%H:%M:%S"
                            ),
                        },
                    )

                else:

                    try:
                        error_data = response.json()
                        error_message = error_data.get(
                            "detail",
                            "The question request failed.",
                        )

                    except ValueError:
                        error_message = (
                            "The server returned "
                            "an invalid error response."
                        )

                    st.error(
                        f"Question failed: {error_message}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the FastAPI backend. "
                    "Please confirm that the API server is running."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The request timed out while generating "
                    "the answer. Please try again."
                )

            except requests.exceptions.RequestException as error:
                st.error(
                    f"An unexpected network error occurred: {error}"
                )

            except Exception as error:
                st.error(
                    f"An unexpected error occurred: {error}"
                )

    # -----------------------------------------------------------
    # Latest answer + full history
    # -----------------------------------------------------------

    history = st.session_state["qa_history"]

    if not history:
        return

    st.divider()

    latest_tab, history_tab = st.tabs(
        ["Latest answer", f"History ({len(history)})"]
    )

    with latest_tab:
        latest = history[0]
        display_answer(
            latest["answer"],
            latest["citations"],
            timestamp=latest["timestamp"],
            key_prefix="latest",
        )

    with history_tab:
        if len(history) == 1:
            st.caption("Only one question asked so far.")
        else:
            for index, entry in enumerate(history):
                st.markdown(
                    f'<div class="history-question">'
                    f'{entry["question"]}</div>',
                    unsafe_allow_html=True,
                )
                display_answer(
                    entry["answer"],
                    entry["citations"],
                    timestamp=entry["timestamp"],
                    key_prefix=f"history_{index}",
                )
                st.divider()


# -------------------------------------------------------------------
# Main application
# -------------------------------------------------------------------

def main():

    render_sidebar()

    # ---------------------------------------------------------------
    # Masthead
    # ---------------------------------------------------------------

    st.markdown(
        """
        <div class="masthead">
            <div class="masthead-kicker">Retrieval-augmented reading</div>
            <div class="masthead-title">Marginalia</div>
            <div class="masthead-subtitle">
                Ask a paper what it means, and it will show you
                exactly where it said so.
            </div>
            <div class="masthead-badges">
                <span class="masthead-badge">BGE embeddings</span>
                <span class="masthead-badge">FAISS retrieval</span>
                <span class="masthead-badge">Cross-encoder reranking</span>
                <span class="masthead-badge">Ollama LLM</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------
    # Upload
    # ---------------------------------------------------------------

    render_upload_section()

    # ---------------------------------------------------------------
    # Question answering
    # ---------------------------------------------------------------

    if st.session_state["uploaded_document"] is not None:

        st.divider()

        render_question_section()

    else:

        st.divider()

        st.info(
            "Your research workspace will appear here "
            "after a document is processed."
        )

    # ---------------------------------------------------------------
    # Footer
    # ---------------------------------------------------------------

    st.markdown(
        """
        <div class="app-footer">
            Marginalia is a local retrieval-augmented reading tool,
            built on FastAPI and Streamlit.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Application entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
