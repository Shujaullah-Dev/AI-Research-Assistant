#  AI Research Assistant

An open-source AI research assistant that uses Retrieval-Augmented Generation (RAG) to help users search, understand, and analyze research papers.

##  Project Status

 Project foundation

The initial backend and frontend infrastructure has been implemented. RAG functionality will be developed incrementally.

##  Planned Features

- PDF document ingestion
- Text extraction and chunking
- Semantic search
- Vector database
- Retrieval-Augmented Generation
- Source citations
- Multi-document question answering
- Research paper summarization
- AI evaluation
- REST API
- Interactive web interface

##  Planned Architecture

PDF → Extraction → Chunking → Embeddings → Vector Database → Retrieval → LLM → Answer + Citations

##  Technology Stack

- Python
- FastAPI
- Streamlit
- PyTorch
- Sentence Transformers
- ChromaDB
- Open-source LLMs
- Docker

## Retrieval-Augmented Generation

The system uses a Retrieval-Augmented Generation (RAG)
pipeline to answer questions about indexed research papers.

The pipeline works as follows:

1. Research papers are converted into text.
2. Text is divided into overlapping chunks.
3. Chunks are converted into semantic embeddings.
4. Embeddings are stored in a FAISS vector index.
5. A user's question is converted into an embedding.
6. FAISS retrieves the most semantically similar chunks.
7. Low-similarity results can be filtered using a configurable threshold.
8. Retrieved chunks are provided to a local LLM through Ollama.
9. The LLM generates a grounded answer.
10. Retrieved document and page information is returned as citations.

The system is designed to reduce unsupported answers by
instructing the LLM to use only retrieved research context.

##  License

MIT