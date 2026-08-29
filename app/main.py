from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="AI Research Assistant",
    description=(
        "An AI-powered research assistant using "
        "Retrieval-Augmented Generation."
    ),
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "AI Research Assistant API is running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }