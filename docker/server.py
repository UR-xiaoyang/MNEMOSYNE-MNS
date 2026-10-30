"""
FastAPI server for MNEMOSYNE-MNS memory ranking service.
Docker-ready production deployment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project scripts to path. Works both from the source tree and /app in Docker.
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "scripts").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from memory_pipeline import create_pipeline
from memory_schema import memory_from_api
from memory_store import MemoryStore
from memory_service import MemoryService
from shadow_mode import ShadowModeCollector
from shadow_runtime import record_retrieval
from citation_detection import detect_citations_in_response
from route_baseline import route_query
from conflict_detection import find_conflicts
from write_gate import proposal_payload
app = FastAPI(
    title="MNEMOSYNE-MNS",
    description="Memory Ranking Service for Agent Memory Systems",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
print("Loading RANK-HYBRID-01 model...")
pipeline = create_pipeline("hybrid")
print("✓ Model loaded successfully")

shadow_collector = None
shadow_enabled = os.environ.get("MNEMOSYNE_SHADOW_MODE", "0") == "1"
if shadow_enabled:
    shadow_collector = ShadowModeCollector(os.environ.get("MNEMOSYNE_SHADOW_DB", str(PROJECT_ROOT / "shadow_traces.db")))
memory_store = MemoryStore(os.environ.get("MNEMOSYNE_DB", str(PROJECT_ROOT / "memory.db")))
memory_service = MemoryService(pipeline, memory_store)


class Memory(BaseModel):
    id: str
    content: str


class RetrievalRequest(BaseModel):
    query: str
    memories: List[Memory]
    top_k: int = Field(default=10, ge=1, le=100)
    scope_level: str = Field(default="global", pattern="^(session|project|global)$")
    destination: str = Field(default="local")
    session_id: str | None = None
    turn_id: str | None = None


class RetrievalResponse(BaseModel):
    results: List[dict]
    latency_ms: float


class MemoryWriteRequest(BaseModel):
    id: str
    content: str
    scope_level: str = Field(default="global", pattern="^(session|project|global)$")
    destination: str = Field(default="local")


class MemoryWriteResponse(BaseModel):
    memory_id: str
    version: int


class MemorySupersedeRequest(BaseModel):
    old_id: str
    new_id: str
    content: str
    scope_level: str = Field(default="global", pattern="^(session|project|global)$")
    destination: str = Field(default="local")


class WriteProposalRequest(BaseModel):
    text: str
    scope_level: str = Field(default="session", pattern="^(session|project|global)$")


class CitationRequest(BaseModel):
    response: str
    injected_memories: List[Memory]


class CitationResponse(BaseModel):
    citations: List[dict]


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "RANK-HYBRID-01",
        "version": "1.0.0"
    }


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(req: RetrievalRequest):
    """
    Retrieve and rank memories for a query.
    
    Returns top-K ranked memories with scores.
    """
    try:
        import time
        start = time.time()
        
        # Convert to MemoryEntity objects
        candidates = [
            memory_from_api(
                memory.id,
                memory.content,
                destination=req.destination,
                scope_level=req.scope_level,
            )
            for memory in req.memories
        ]
        
        # Retrieve and rank
        results = pipeline.retrieve_and_rank(
            query_text=req.query,
            candidate_memories=candidates,
            scope_level=req.scope_level,
            destination=req.destination,
            top_k=req.top_k
        )
        
        # Assemble context
        assembled = pipeline.assemble_context(results, max_tokens=4000)
        record_retrieval(
            shadow_collector,
            enabled=shadow_enabled and req.session_id is not None and req.turn_id is not None,
            session_id=req.session_id or "",
            turn_id=req.turn_id or "",
            query={"text": req.query, "scope": {"level": req.scope_level}, "destination": req.destination},
            candidates=results,
            metadata={"ranker_model_id": pipeline.ranker.model_id, "injected_count": len(assembled)},
        )

        # Map back to original IDs
        response_results = []
        for result in assembled:
            response_results.append({
                "memory_id": result["memory_id"],
                "content": result["content"],
                "rank": result["rank"],
                "score": result["score"],
                "injected": result["injected"],
                "metadata": result.get("metadata", {})
            })
        
        latency = (time.time() - start) * 1000
        
        return RetrievalResponse(
            results=response_results,
            latency_ms=round(latency, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories", response_model=MemoryWriteResponse)
async def write_memory(req: MemoryWriteRequest):
    memory = memory_service.write(
        req.id,
        req.content,
        scope_level=req.scope_level,
        destination=req.destination,
    )
    return MemoryWriteResponse(memory_id=memory.memory_id, version=memory.version)


@app.post("/memories/supersede", response_model=MemoryWriteResponse)
async def supersede_memory(req: MemorySupersedeRequest):
    try:
        updated = memory_service.supersede(
            req.old_id,
            req.new_id,
            req.content,
            scope_level=req.scope_level,
            destination=req.destination,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MemoryWriteResponse(memory_id=updated.memory_id, version=updated.version)


@app.post("/retrieve_store", response_model=RetrievalResponse)
async def retrieve_store(req: RetrievalRequest):
    import time
    start = time.time()
    outcome = memory_service.retrieve_store(
        req.query,
        scope_level=req.scope_level,
        destination=req.destination,
        top_k=req.top_k,
    )
    return RetrievalResponse(
        results=outcome["assembled"],
        latency_ms=round((time.time() - start) * 1000, 2),
    )


@app.post("/conflicts")
async def conflicts(req: RetrievalRequest):
    memories = [
        memory_from_api(item.id, item.content, destination=req.destination, scope_level=req.scope_level)
        for item in req.memories
    ]
    return {
        "conflicts": [
            {"left_id": item.left_id, "right_id": item.right_id, "reason": item.reason, "confidence": item.confidence}
            for item in find_conflicts(memories)
        ]
    }


@app.post("/route")
async def route(req: WriteProposalRequest):
    decision = route_query(req.text)
    return {
        "primary_path": decision.primary_path,
        "candidate_paths": decision.candidate_paths,
        "confidence_proxy": decision.confidence_proxy,
        "force_slow_fallback": decision.force_slow_fallback,
        "fallback_reason": decision.fallback_reason,
        "query_kind": decision.query_kind,
        "requires_high_authority": decision.requires_high_authority,
    }


@app.post("/write_proposals")
async def write_proposals(req: WriteProposalRequest):
    """Return write suggestions; this endpoint never commits them."""
    return {"proposals": proposal_payload(req.text, scope_level=req.scope_level)}


@app.post("/detect_citations", response_model=CitationResponse)
async def detect_citations(req: CitationRequest):
    """
    Detect which memories were cited in a model response.
    """
    try:
        injected = [
            {
                "memory_id": m.id,
                "content": m.content,
                "rank": i + 1
            }
            for i, m in enumerate(req.injected_memories)
        ]
        
        citations = detect_citations_in_response(req.response, injected)
        
        return CitationResponse(citations=citations)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def list_models():
    """List available ranking models."""
    return {
        "available_models": ["hybrid", "e5", "tree"],
        "current_model": "hybrid",
        "model_info": {
            "model_id": "MNEMOSYNE-MNS-RANK-HYBRID-01",
            "ndcg@10": 0.5796,
            "recall@1": 0.4071,
            "features": "E5 + 12 structured features"
        }
    }


@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "service": "MNEMOSYNE-MNS",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "retrieve": "/retrieve",
            "memories": "/memories",
            "memories_supersede": "/memories/supersede",
            "retrieve_store": "/retrieve_store",
            "route": "/route",
            "conflicts": "/conflicts",
            "write_proposals": "/write_proposals",
            "detect_citations": "/detect_citations",
            "models": "/models",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
