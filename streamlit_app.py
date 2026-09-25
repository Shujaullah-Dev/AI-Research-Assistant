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
# Custom styling
#
# Design language: a reading room / lab notebook for papers.
# Serif for anything read (headings, questions, answers), a plain
# grotesque for anything operated (buttons, labels, chrome).
# Ink-on-paper palette with a single warm accent used sparingly.
# -------------------------------------------------------------------

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400;1,6..72,500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --ink: #1C2230;
        --ink-soft: #5B6472;
        --paper: #FAF9F4;
        --paper-raised: #FFFFFF;
        --line: #DEDACB;
        --accent: #A8672A;
        --accent-soft: rgba(168, 103, 42, 0.10);
        --teal: #2E5C55;
        --teal-soft: rgba(46, 92, 85, 0.10);
        --brick: #9A4433;
        --brick-soft: rgba(154, 68, 51, 0.10);
        --font-serif: 'Newsreader', Georgia, serif;
        --font-sans: 'IBM Plex Sans', -apple-system, sans-serif;
    }

    html, body, [class*="css"] {
        font-family: var(--font-sans);
        color: var(--ink);
    }

    .stApp {
        background: var(--paper);
    }

    /* -------------------------------------------------------------
       Global layout
    ------------------------------------------------------------- */

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        max-width: 1180px;
    }

    /* -------------------------------------------------------------
       Masthead
    ------------------------------------------------------------- */

    @keyframes settle {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .masthead {
        padding: 0.75rem 0 1.4rem 0;
        border-bottom: 2px solid var(--ink);
        margin-bottom: 0.5rem;
        animation: settle 0.5s ease-out;
    }

    .masthead-kicker {
        font-family: var(--font-sans);
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--accent);
        margin-bottom: 0.35rem;
    }

    .masthead-title {
        font-family: var(--font-serif);
        font-size: 2.75rem;
        font-weight: 500;
        line-height: 1.1;
        letter-spacing: -0.01em;
        margin-bottom: 0.4rem;
    }

    .masthead-subtitle {
        font-family: var(--font-serif);
        font-style: italic;
        font-size: 1.15rem;
        color: var(--ink-soft);
        max-width: 46ch;
    }

    /* -------------------------------------------------------------
       Section headers
    ------------------------------------------------------------- */

    .section-title {
        font-family: var(--font-serif);
        font-size: 1.55rem;
        font-weight: 500;
        margin-top: 0.5rem;
        margin-bottom: 0.15rem;
        color: var(--ink);
    }

    .section-description {
        color: var(--ink-soft);
        font-size: 0.95rem;
        margin-bottom: 1.1rem;
        max-width: 62ch;
    }

    /* -------------------------------------------------------------
       Document statistics cards
    ------------------------------------------------------------- */

    .stat-card {
        padding: 0.9rem 1.1rem;
        border-top: 2px solid var(--ink);
        background: var(--paper-raised);
        min-height: 92px;
    }

    .stat-label {
        font-size: 0.78rem;
        color: var(--ink-soft);
        margin-bottom: 0.3rem;
    }

    .stat-value {
        font-family: var(--font-serif);
        font-size: 1.3rem;
        font-weight: 500;
        word-break: break-word;
        color: var(--ink);
    }

    /* -------------------------------------------------------------
       Answer container
    ------------------------------------------------------------- */

    .answer-container {
        padding: 1.5rem 1.6rem;
        border-left: 3px solid var(--accent);
        background: var(--paper-raised);
        margin-top: 0.75rem;
        margin-bottom: 1.1rem;
    }

    .answer-label {
        font-family: var(--font-sans);
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }

    .answer-timestamp {
        font-size: 0.78rem;
        color: var(--ink-soft);
        margin-bottom: 0.9rem;
    }

    .answer-body {
        font-family: var(--font-serif);
        font-size: 1.12rem;
        line-height: 1.65;
        color: var(--ink);
    }

    /* -------------------------------------------------------------
       History question header
    ------------------------------------------------------------- */

    .history-question {
        font-family: var(--font-serif);
        font-style: italic;
        font-size: 1.1rem;
        color: var(--ink);
        margin-bottom: 0.5rem;
        padding-top: 0.75rem;
    }

    /* -------------------------------------------------------------
       Citation / footnote cards
    ------------------------------------------------------------- */

    .citation-card {
        padding: 0.7rem 0 0.7rem 0.9rem;
        border-left: 2px solid var(--line);
        margin-bottom: 0.6rem;
    }

    .citation-title {
        font-family: var(--font-serif);
        font-size: 1.02rem;
        color: var(--ink);
        margin-bottom: 0.2rem;
    }

    .citation-number {
        color: var(--accent);
        font-style: italic;
        margin-right: 0.15rem;
    }

    .citation-meta {
        font-size: 0.82rem;
        color: var(--ink-soft);
        display: flex;
        gap: 1.1rem;
        flex-wrap: wrap;
    }

    /* -------------------------------------------------------------
       Sidebar
    ------------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: var(--paper-raised);
        border-right: 1px solid var(--line);
    }

    .sidebar-brand {
        font-family: var(--font-serif);
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 0.1rem;
        color: var(--ink);
    }

    .sidebar-brand-rule {
        width: 2.5rem;
        height: 2px;
        background: var(--accent);
        margin-bottom: 0.7rem;
    }

    .sidebar-description {
        font-size: 0.88rem;
        color: var(--ink-soft);
        line-height: 1.55;
    }

    .sidebar-item {
        font-size: 0.9rem;
        margin-bottom: 0.55rem;
        color: var(--ink);
    }

    .sidebar-item strong {
        color: var(--accent);
        font-family: var(--font-serif);
        font-style: italic;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        border-radius: 3px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
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
    }

    /* -------------------------------------------------------------
       Footer
    ------------------------------------------------------------- */

    .app-footer {
        text-align: left;
        padding-top: 1.5rem;
        margin-top: 3rem;
        border-top: 1px solid var(--line);
        color: var(--ink-soft);
        font-size: 0.85rem;
    }

    /* -------------------------------------------------------------
       Streamlit widget overrides
    ------------------------------------------------------------- */

    .stButton > button {
        border-radius: 3px;
        font-weight: 500;
        min-height: 2.7rem;
        border: 1px solid var(--ink);
        background: var(--ink);
        color: var(--paper);
        transition: background 0.15s ease, color 0.15s ease;
    }

    .stButton > button:hover {
        background: var(--paper);
        color: var(--ink);
        border: 1px solid var(--ink);
    }

    .stButton > button:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }

    .stDownloadButton > button {
        border-radius: 3px;
        border: 1px solid var(--line);
        background: var(--paper);
        color: var(--ink);
        font-weight: 500;
    }

    .stDownloadButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    [data-testid="stFileUploader"] {
        border-radius: 4px;
    }

    .stTextArea textarea {
        font-family: var(--font-serif);
        font-size: 1.05rem;
        border-radius: 3px;
        border: 1px solid var(--line);
    }

    .stTextArea textarea:focus {
        border-color: var(--accent);
        box-shadow: none;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-serif);
        font-size: 1.02rem;
        color: var(--ink-soft);
    }

    .stTabs [aria-selected="true"] {
        color: var(--ink) !important;
    }

    hr {
        border-color: var(--line) !important;
    }

    ::selection {
        background: var(--accent-soft);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Session state initialization
# -------------------------------------------------------------------

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

    response = requests.post(
        UPLOAD_ENDPOINT,
        files=files,
        timeout=300,
    )

    return response


def ask_question(question):
    """
    Send a question to the FastAPI backend.
    """

    payload = {
        "question": question,
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

        st.divider()

        # ---------------------------------------------------------
        # Backend status
        # ---------------------------------------------------------

        st.markdown("**Backend**")

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

        st.markdown("**How it works**")

        st.markdown(
            """
            <div class="sidebar-item"><strong>1.</strong> Upload a research paper</div>
            <div class="sidebar-item"><strong>2.</strong> Extract and index the content</div>
            <div class="sidebar-item"><strong>3.</strong> Ask a question</div>
            <div class="sidebar-item"><strong>4.</strong> Retrieve relevant passages</div>
            <div class="sidebar-item"><strong>5.</strong> Generate a grounded answer</div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown("**System**")

        st.caption("FastAPI backend")
        st.caption("BGE embeddings")
        st.caption("FAISS retrieval")
        st.caption("Cross-encoder reranking")
        st.caption("Ollama LLM")

        # ---------------------------------------------------------
        # Session controls
        # ---------------------------------------------------------

        if st.session_state["uploaded_document"] is not None:

            st.divider()

            st.markdown("**Session**")

            st.caption(
                f"{len(st.session_state['qa_history'])} question(s) asked"
            )

            if st.button(
                "Start new session",
                use_container_width=True,
                help="Clear the current document and question history",
            ):
                st.session_state["uploaded_document"] = None
                st.session_state["last_answer"] = None
                st.session_state["last_citations"] = []
                st.session_state["qa_history"] = []
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
                    <span>Page {page_number}</span>
                    <span>Chunk {chunk_id}</span>
                    <span>{score_text}</span>
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