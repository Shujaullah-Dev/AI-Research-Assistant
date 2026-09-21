import requests
import streamlit as st


# -------------------------------------------------------------------
# Application configuration
# -------------------------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000"

UPLOAD_ENDPOINT = (
    f"{API_BASE_URL}/documents/upload"
)

ASK_ENDPOINT = (
    f"{API_BASE_URL}/ask"
)

MAX_UPLOAD_SIZE = 20 * 1024 * 1024


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide",
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


# -------------------------------------------------------------------
# Display functions
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
        "The research paper was uploaded successfully."
    )

    st.subheader("📄 Document Information")

    information_col1, information_col2 = st.columns(2)

    with information_col1:
        st.write("**Document:**")
        st.write(document_name)

        st.write("**Pages processed:**")
        st.write(pages_extracted)

        st.write("**Chunks created:**")
        st.write(chunks_indexed)

    with information_col2:
        st.write("**Processing status:**")
        st.write(status)

        st.write("**Vector store:**")
        st.write(vector_store)

    with st.expander("Embedding Information"):
        st.write(
            f"**Embedding model:** {embedding_model}"
        )


def display_citations(citations):
    """
    Display grounded citations returned by the API.
    """

    if not citations:
        st.info(
            "No citations were returned for this answer."
        )

        return

    st.subheader("📚 Sources and Citations")

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

        with st.expander(
            f"Source {citation_number}: "
            f"{document_name} "
            f"(Page {page_number})"
        ):
            st.write(
                f"**Document:** {document_name}"
            )

            st.write(
                f"**Page:** {page_number}"
            )

            st.write(
                f"**Chunk ID:** {chunk_id}"
            )

            if score is not None:
                st.write(
                    f"**Relevance score:** {score:.4f}"
                )


def display_answer(answer, citations):
    """
    Display the generated answer and its citations.
    """

    st.subheader("🤖 Generated Answer")

    if not answer or not answer.strip():
        st.warning(
            "The system did not return an answer."
        )

        return

    st.write(answer)

    st.divider()

    display_citations(citations)


# -------------------------------------------------------------------
# Upload section
# -------------------------------------------------------------------

def render_upload_section():
    """
    Render the PDF upload and processing interface.
    """

    st.header("📄 Upload Research Paper")

    st.write(
        "Upload a PDF file to extract its content, "
        "create embeddings, and prepare it for questions."
    )

    uploaded_file = st.file_uploader(
        "Choose a research paper",
        type=["pdf"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info(
            "Please upload a PDF research paper to begin."
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
        f"Selected file: {file_name} "
        f"({file_size_mb:.2f} MB)"
    )

    upload_button = st.button(
        "Upload and Process PDF",
        type="primary",
        use_container_width=True,
    )

    if upload_button:
        with st.spinner(
            "Uploading and processing your research paper..."
        ):
            try:
                response = upload_document(
                    uploaded_file
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

                else:
                    try:
                        error_data = response.json()

                        error_message = error_data.get(
                            "detail",
                            "The upload request failed.",
                        )

                    except ValueError:
                        error_message = (
                            "The server returned an invalid error response."
                        )

                    st.error(
                        f"Upload failed: {error_message}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the FastAPI backend. "
                    "Please confirm that the API server is running."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The request timed out while processing the PDF. "
                    "Please try again."
                )

            except requests.exceptions.RequestException as error:
                st.error(
                    f"An unexpected network error occurred: {error}"
                )

            except Exception as error:
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

    st.header("💬 Ask Questions")

    st.write(
        "Ask a question about the uploaded research paper."
    )

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What dataset did the researchers use?"
        ),
        height=100,
    )

    ask_button = st.button(
        "Ask Question",
        type="primary",
        use_container_width=True,
    )

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

                else:
                    try:
                        error_data = response.json()

                        error_message = error_data.get(
                            "detail",
                            "The question request failed.",
                        )

                    except ValueError:
                        error_message = (
                            "The server returned an invalid error response."
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
                    "The request timed out while generating the answer. "
                    "Please try again."
                )

            except requests.exceptions.RequestException as error:
                st.error(
                    f"An unexpected network error occurred: {error}"
                )

            except Exception as error:
                st.error(
                    f"An unexpected error occurred: {error}"
                )

    if st.session_state["last_answer"] is not None:
        display_answer(
            st.session_state["last_answer"],
            st.session_state["last_citations"],
        )


# -------------------------------------------------------------------
# Main application
# -------------------------------------------------------------------

def main():
    st.title("📚 AI Research Assistant")

    st.write(
        "Upload a research paper and ask questions "
        "using retrieval-augmented generation."
    )

    st.divider()

    render_upload_section()

    st.divider()

    if st.session_state["uploaded_document"] is not None:
        render_question_section()

    else:
        st.info(
            "Upload and process a research paper "
            "before asking questions."
        )


if __name__ == "__main__":
    main()