from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from core.indexer import DocumentIndexer
from core.retriever import RAGEngine
import os

router = APIRouter()

# Initialize the engine with HNSW configuration for 10k+ embeddings
# efConstruction: trade-off between index build time and search accuracy
# M: number of bi-directional links created for every new element
indexer = DocumentIndexer(index_type="HNSW", M=32, ef_construction=200)
rag_engine = RAGEngine(indexer=indexer)

@router.post("/ingest")
async def ingest_documents(file: UploadFile = File(...)):
    """
    Endpoint to process and index documents. 
    Uses environment isolation for API keys.
    """
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="API Key isolation error.")

    try:
        # Save briefly to temp, then indexer will process and middleware will delete
        temp_path = f"./temp_uploads/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        stats = indexer.add_to_index(temp_path)
        return {"status": "success", "embeddings_created": stats["count"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_documents(query: str):
    """
    Sub-second retrieval using HNSW similarity search.
    """
    context = rag_engine.get_relevant_context(query)
    answer = rag_engine.generate_answer(query, context)
    return {"answer": answer, "context_fragments": context}