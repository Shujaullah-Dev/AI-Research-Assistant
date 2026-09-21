
# AI Research Assistant

An open-source AI research assistant that uses Retrieval-Augmented Generation (RAG) to help users search, understand, and analyze research papers.

The system accepts research-paper PDFs, extracts and chunks their text, creates semantic embeddings, retrieves relevant passages, reranks them using a cross-encoder, and generates grounded answers using a local LLM through Ollama.

## Project Status

**Working prototype**

The core RAG pipeline and REST API are implemented and tested.

Current capabilities include:

- PDF document upload and ingestion
- Page-level text extraction
- Overlapping text chunking
- BGE semantic embeddings
- FAISS vector search
- Cross-encoder reranking
- Retrieval relevance filtering
- Grounded question answering
- Source citations
- Multi-document retrieval
- REST API with FastAPI
- Automated testing and retrieval evaluation

## Architecture

```text
PDF
 │
 ▼
Document Ingestion
 │
 ▼
Text Extraction
 │
 ▼
Chunking
 │
 ▼
BGE Embeddings
 │
 ▼
FAISS Vector Store
 │
 ▼
Semantic Retrieval
 │
 ▼
Cross-Encoder Reranking
 │
 ▼
Relevance Filtering
 │
 ▼
Local LLM (Ollama)
 │
 ▼
Grounded Answer + Citations
```

## How RAG Works

The system uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions about indexed research papers.

The pipeline works as follows:

1.  A research-paper PDF is uploaded to the API. 
2.  The PDF is converted into page-level text. 
3.  Text is divided into overlapping chunks. 
4.  Each chunk is converted into a semantic embedding using a BGE embedding model. 
5.  Embeddings are stored in a FAISS vector index. 
6.  A user's question is converted into an embedding. 
7.  FAISS retrieves candidate chunks based on semantic similarity. 
8.  A cross-encoder reranker reorders the retrieved candidates by relevance. 
9.  Low-relevance results can be filtered. 
10.  The most relevant context is provided to a local LLM through Ollama. 
11.  The LLM generates an answer grounded in the retrieved context. 
12.  Document and page information is returned as citations. 

The system is designed to reduce unsupported answers by instructing the LLM to rely only on retrieved research context. When relevant information cannot be found in the indexed documents, the system can return an insufficient-context response instead of fabricating an answer.

## Technology Stack

- **Python** — application and RAG pipeline 
- **FastAPI** — REST API 
- **PyMuPDF** — PDF text extraction 
- **Sentence Transformers** — semantic embeddings and reranking 
- **BAAI/bge-small-en-v1.5** — embedding model 
- **FAISS** — vector similarity search 
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — reranking 
- **Ollama** — local LLM runtime 
- **llama3.2:3b** — local language model 
- **Pydantic** — request/response validation 
- **Pytest** — automated testing 

## API

The application exposes a REST API through FastAPI.

### Start the API

From the project root:


```
python -m uvicorn app.main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```
http://127.0.0.1:8000/docs
```

### Health Check

```
GET /health
```

Example response:

```
{
  "status": "healthy"
}
```

### Upload a Research Paper

```
POST /documents/upload
```

The endpoint accepts PDF files and:

-  validates the file type 
-  rejects empty files 
-  enforces a 20 MB upload limit 
-  extracts page text 
-  creates chunks 
-  generates BGE embeddings 
-  updates the FAISS vector store 

### Ask a Question

```
POST /ask
```

Example request:

```
{
  "question": "What dataset was used to train the model?",
  "top_k": 5,
  "final_k": 5
}
```

The response contains the generated answer together with source citations.

Example:

```
{
  "answer": "The researchers trained the model using the CIFAR-10 dataset.",
  "citations": [
    {
      "source_id": "0",
      "document_name": "demo_paper.pdf",
      "page_number": 1,
      "chunk_id": 0,
      "score": 0.8
    }
  ]
}
```

## Retrieval and Reranking

The retrieval system uses a two-stage approach.

### Stage 1 — Semantic Retrieval

The BGE embedding model converts both documents and questions into vectors.

FAISS then performs similarity search to retrieve the most relevant candidate chunks.

### Stage 2 — Cross-Encoder Reranking

The retrieved candidates are passed to a cross-encoder:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The cross-encoder evaluates the relationship between the user's question and each retrieved chunk and produces a relevance score.

This allows the system to improve the ordering of retrieved context before sending it to the LLM.

## Grounded Citations

Answers include citation metadata associated with the retrieved chunks.

Each citation can contain:

-  document name 
-  page number 
-  chunk ID 
-  retrieval/reranking score 

This provides traceability from an answer back to the source material used to generate it.

## Retrieval Evaluation

The project includes evaluation scripts and tests for measuring retrieval quality.

The retrieval evaluation tracks metrics such as:

-  Top-1 accuracy 
-  Top-3 accuracy 
-  Mean Reciprocal Rank (MRR) 

The project also includes evaluation of the cross-encoder reranking stage.

These evaluations are used to compare retrieval behavior and identify cases where semantically related but incorrect chunks are retrieved.

## Testing

Run the complete test suite with:

```
python -m pytest
```

The project currently includes tests covering:

-  API endpoints 
-  PDF loading 
-  text chunking 
-  embeddings 
-  vector-store behavior 
-  retrieval 
-  RAG pipeline behavior 
-  citation generation 
-  relevance filtering 
-  retrieval evaluation 
-  reranking evaluation 

## Project Structure

```
AI-Research-Assistant/
│
├── app/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── chunking/
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   └── service.py
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   ├── schemas.py
│   │   └── service.py
│   │
│   ├── rag/
│   │   └── pipeline.py
│   │
│   ├── retrieval/
│   │   └── reranker.py
│   │
│   ├── vector_store/
│   │   └── faiss_store.py
│   │
│   └── main.py
│
├── scripts/
│   ├── build_bge_index.py
│   ├── check_retrieval.py
│   ├── check_reranking.py
│   └── run_rag_demo.py
│
├── tests/
│   ├── test_api.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_pdf_loader.py
│   ├── test_rag_pipeline.py
│   ├── test_retriever.py
│   ├── test_vector_store.py
│   ├── test_citations.py
│   ├── test_relevance_filtering.py
│   ├── test_retrieval_evaluation.py
│   └── test_reranking_evaluation.py
│
├── data/
│   └── .gitkeep
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Local Development

Create and activate a virtual environment:

```
python -m venv .venv
```

Windows PowerShell:

```
.venv\Scripts\Activate.ps1
```

Install dependencies:

```
pip install -r requirements.txt
```

Ollama must also be installed and the required local model must be available.

The application currently uses:
```
llama3.2:3b
```

After starting Ollama, start the FastAPI application:

```
python -m uvicorn app.main:app --reload
```

## Data and Generated Files

Generated application data is intentionally excluded from Git.

This includes:

-  uploaded PDFs 
-  generated embeddings 
-  FAISS indexes 
-  vector-store metadata 

These files are stored under the `data/` directory during local development and are ignored by Git.

## Current Limitations

This is currently a research-oriented prototype.

Some planned capabilities are not yet implemented, including:

-  dedicated interactive web UI 
-  advanced research-paper summarization 
-  comprehensive multi-user document management 
-  production deployment 
-  large-scale evaluation datasets

