from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv
import uvicorn
import shutil
from datetime import datetime
import uuid

# Load environment
load_dotenv()

# Import your modules
import sys
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from finance_insight_lite.modules.processor import load_documents_fastest, pdf_to_documents
from finance_insight_lite.modules.verctor_store import build_vector_db
from finance_insight_lite.modules.rag_agent import FinancialRAGAgent

# Initialize FastAPI
app = FastAPI(
    title="Finance Insight Lite API",
    description="Advanced RAG API for financial document analysis",
    version="1.0.0"
)

# CORS middleware - allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use Redis or database)
class AppState:
    agent: Optional[Any] = None
    vector_db: Optional[Any] = None
    document_info: Dict[str, Any] = {}

state = AppState()


# ============================================================================
# MODELS
# ============================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    use_self_rag: bool = Field(default=True, description="Enable Self-RAG verification")
    max_retries: int = Field(default=1, ge=0, le=3, description="Max retries for Self-RAG")


class QueryResponse(BaseModel):
    answer: str
    source_pages: List[Any]
    confidence: str
    relevant_docs_count: int
    verification: Optional[Dict[str, Any]] = None
    processing_time_ms: float


class DocumentInfo(BaseModel):
    filename: str
    pages: int
    processed_at: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    agent_initialized: bool
    document_loaded: bool
    timestamp: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """API root endpoint"""
    return {
        "message": "Finance Insight Lite API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and status"""
    return HealthResponse(
        status="healthy",
        agent_initialized=state.agent is not None,
        document_loaded=state.vector_db is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/upload", response_model=DocumentInfo, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    use_self_rag: bool = True
):
    """
    Upload and process a PDF, Excel, CSV, image, or supported financial document
    
    This endpoint:
    1. Saves the uploaded PDF
    2. Processes it into chunks
    3. Creates vector embeddings
    4. Initializes the RAG agent
    """
    
    supported_extensions = (".pdf", ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg")
    if not file.filename or not file.filename.lower().endswith(supported_extensions):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    try:
        # Save uploaded file
        upload_dir = Path("data/uploaded")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        load_result = load_documents_fastest(str(file_path), max_workers=1)
        documents = load_result["documents"]
        
        # Build vector database
        db_path = Path("data/vector_db")
        db_path.mkdir(parents=True, exist_ok=True)
        
        state.vector_db, chunks = build_vector_db(
            documents,
            db_path=str(db_path),
            source_paths=[str(file_path)],
        )
        
        # Create agent
        state.agent = FinancialRAGAgent(state.vector_db, chunks=chunks)
        
        # Store document info
        state.document_info = {
            "filename": file.filename,
            "pages": len(documents),
            "processed_at": datetime.now().isoformat(),
            "chunks": len(chunks)
        }
        
        return DocumentInfo(**state.document_info)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_document(request: QueryRequest):
    """
    Query the financial document using RAG
    
    Requires a document to be uploaded first via /upload endpoint
    """
    
    if state.agent is None:
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please upload a document first using /upload endpoint"
        )
    
    try:
        import time
        start_time = time.time()
        
        # Process query
        result = state.agent.process_query(
            query=request.question,
            use_self_rag=request.use_self_rag,
            max_retries=request.max_retries,
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return QueryResponse(
            answer=result['answer'],
            source_pages=result['source_pages'],
            confidence=result['confidence'],
            relevant_docs_count=result['relevant_docs_count'],
            verification=result.get('verification'),
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/document/info", response_model=DocumentInfo, tags=["Documents"])
async def get_document_info():
    """Get information about the currently loaded document"""
    
    if not state.document_info:
        raise HTTPException(status_code=400, detail="No document loaded")
    
    return DocumentInfo(**state.document_info)


@app.delete("/document", tags=["Documents"])
async def clear_document():
    """Clear the currently loaded document and reset the agent"""
    
    state.agent = None
    state.vector_db = None
    state.document_info = {}
    
    return {"message": "Document cleared successfully"}


@app.get("/sample-questions", tags=["Query"])
async def get_sample_questions():
    """Get sample questions for the financial document"""
    
    return {
        "questions": [
            "What is the net income for Q3 2025?",
            "What is the free cash flow?",
            "What is the gearing ratio?",
            "How much was the dividend declared?",
            "What are the key financial highlights?",
            "What is the adjusted EBITDA?",
            "What are the capital expenditures?",
            "What is the debt-to-equity ratio?"
        ]
    }


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Finance Insight Lite API starting...")
    print(f"📄 API Docs: http://localhost:8000/docs")
    
    # Try to load default document if exists
    default_pdf = Path("data/raw/saudi-aramco-q3-2025-interim-report-english.pdf")
    if default_pdf.exists():
        try:
            print(f"📂 Loading default document: {default_pdf.name}")
            documents = pdf_to_documents(str(default_pdf))
            state.vector_db, chunks = build_vector_db(
                documents,
                db_path="data/vector_db",
                source_paths=[str(default_pdf)],
            )
            state.agent = FinancialRAGAgent(state.vector_db, chunks=chunks)
            state.document_info = {
                "filename": default_pdf.name,
                "pages": len(documents),
                "processed_at": datetime.now().isoformat(),
                "chunks": len(chunks)
            }
            print(f"✅ Default document loaded: {len(documents)} pages")
        except Exception as e:
            print(f"⚠️ Could not load default document: {e}")
    
    print("✅ API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down Finance Insight Lite API...")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
