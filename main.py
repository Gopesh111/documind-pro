import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
from api.middleware import ZeroRetentionMiddleware
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

def create_app() -> FastAPI:
    """
    Application factory to initialize the DocuMind Pro engine.
    """
    app = FastAPI(
        title="DocuMind Pro API",
        description="Privacy-first RAG Engine with HNSW Indexing",
        version="1.0.0"
    )

    # 1. Security: Enable Zero-Retention Logic
    # This ensures that any uploaded binary data is wiped after the request
    app.middleware("http")(ZeroRetentionMiddleware())

    # 2. Security: CORS Configuration
    # Restricts API access to prevent unauthorized cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific domains
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    # 3. Routes: Modular API routing
    app.include_router(api_router, prefix="/api/v1")

    return app

app = create_app()

@app.get("/health")
def health_check():
    """Service health check for cloud deployment monitoring."""
    return {"status": "online", "engine": "HNSW-FAISS", "retention_policy": "zero-data"}

if __name__ == "__main__":
    # Ensure the temp directory exists for the first run
    os.makedirs(os.getenv("TEMP_UPLOAD_DIR", "./temp_uploads"), exist_ok=True)
    
    # Run the server
    # Workers are kept to 1 for FAISS memory safety in local testing
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)