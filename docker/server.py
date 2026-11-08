"""
FastAPI server for MNEMOSYNE-MNS memory ranking service.
Docker-ready production deployment.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from context_memory import ContextMemoryService
from shadow_mode import ShadowModeCollector
from shadow_runtime import record_citation, record_correction, record_retrieval
from api_security import authorization_is_valid
from citation_detection import detect_citations_in_response
from route_baseline import route_query
from conflict_detection import find_conflicts
from write_gate import proposal_payload
from payload_filter import filter_turn_payload
from memory_mode import get_memory_mode, is_experimental_global_enabled
from experimental_global_memory import ExperimentalGlobalMemory
app = FastAPI(
    title="MNEMOSYNE-MNS",
    description="Memory Ranking Service for Agent Memory Systems",
    version="1.0.0"
)

_api_key = os.environ.get("MNEMOSYNE_API_KEY", "")


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.url.path != "/health" and not authorization_is_valid(_api_key, request.headers.get("Authorization")):
        return JSONResponse(status_code=401, content={"detail": "invalid or missing bearer token"})
    return await call_next(request)


cors_origins = [origin.strip() for origin in os.environ.get("MNEMOSYNE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Initialize pipeline - Use V3 by default, fallback to hybrid
ranker_version = os.environ.get("MNEMOSYNE_RANKER_VERSION", "v3")
print(f"Loading RANK model (version: {ranker_version})...")
try:
    pipeline = create_pipeline(ranker_version)
    print(f"✓ Model loaded successfully: {ranker_version.upper()}")
except Exception as e:
    print(f"✗ Failed to load {ranker_version}: {e}")
    if ranker_version != "hybrid":
        print("✓ Falling back to hybrid baseline...")
        pipeline = create_pipeline("hybrid")
    else:
        raise

# Get memory mode
try:
    memory_mode = get_memory_mode()
    print(f"✓ Memory mode: {memory_mode}")
except ValueError as e:
    print(f"✗ Invalid MNEMOSYNE_MEMORY_MODE: {e}")
    print("✓ Falling back to session mode")
    memory_mode = "session"

shadow_collector = None
shadow_enabled = os.environ.get("MNEMOSYNE_SHADOW_MODE", "0") == "1"
if shadow_enabled:
    shadow_collector = ShadowModeCollector(os.environ.get("MNEMOSYNE_SHADOW_DB", str(PROJECT_ROOT / "shadow_traces.db")))
memory_store = MemoryStore(os.environ.get("MNEMOSYNE_DB", str(PROJECT_ROOT / "memory.db")))
memory_service = MemoryService(pipeline, memory_store)
context_service = ContextMemoryService(memory_store)
experimental_global = ExperimentalGlobalMemory(memory_store) if is_experimental_global_enabled() else None


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
    project_id: str = ""


class RetrievalResponse(BaseModel):
    results: List[dict]
    latency_ms: float
    retrieval_event_id: str | None = None


class MemoryWriteRequest(BaseModel):
    id: str
    content: str
    scope_level: str = Field(default="global", pattern="^(session|project|global)$")
    destination: str = Field(default="local")
    session_id: str | None = None
    project_id: str = ""


class MemoryWriteResponse(BaseModel):
    memory_id: str
    version: int


class MemorySupersedeRequest(BaseModel):
    old_id: str
    new_id: str
    content: str
    scope_level: str = Field(default="global", pattern="^(session|project|global)$")
    destination: str = Field(default="local")
    session_id: str | None = None
    project_id: str = ""


class WriteProposalRequest(BaseModel):
    text: str
    scope_level: str = Field(default="session", pattern="^(session|project|global)$")


class ContextTurnRequest(BaseModel):
    session_id: str
    turn_id: str
    project_id: str = ""
    user_text: str
    assistant_text: str
    destination: str = Field(default="local")


class ContextTurnResponse(BaseModel):
    memory_id: str


class ContextCheckpointRequest(BaseModel):
    session_id: str
    turn_id: str
    project_id: str = ""
    transcript: str
    destination: str = Field(default="local")


class ContextCheckpointResponse(BaseModel):
    memory_id: str


class CitationMemory(BaseModel):
    id: str = ""
    memory_id: str = ""
    content: str

    @property
    def resolved_id(self) -> str:
        return self.id or self.memory_id


class CitationRequest(BaseModel):
    response: str
    injected_memories: List[CitationMemory]


class CitationResponse(BaseModel):
    citations: List[dict]


class CitationLogRequest(BaseModel):
    session_id: str
    turn_id: str
    retrieval_event_id: str
    memory_id: str
    citation: dict
    original_rank: int
    context_position: int | None = None


class CorrectionLogRequest(BaseModel):
    session_id: str
    turn_id: str
    citation_event_id: str
    memory_id: str
    correction_type: str
    feedback: dict


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": pipeline.ranker.model_id if hasattr(pipeline, 'ranker') else "UNKNOWN",
        "version": "1.0.0",
        "ranker_version": ranker_version
    }


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(req: RetrievalRequest):
    """
    Retrieve and rank memories for a query.
    
    Returns top-K ranked memories with scores.
    
    Legacy endpoint - only available in experimental_global mode.
    """
    if memory_mode == "session":
        raise HTTPException(
            status_code=409,
            detail="retrieve endpoint is disabled in session mode; use /context/recent instead"
        )
    
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
            top_k=req.top_k,
            session_id=req.session_id,
            project_id=req.project_id or None,
        )
        
        # Assemble context
        assembled = pipeline.assemble_context(results, max_tokens=4000)
        retrieval_event_id = record_retrieval(
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
            latency_ms=round(latency, 2),
            retrieval_event_id=retrieval_event_id,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories", response_model=MemoryWriteResponse)
async def write_memory(req: MemoryWriteRequest):
    try:
        memory = memory_service.write(
            req.id,
            req.content,
            scope_level=req.scope_level,
            destination=req.destination,
            session_id=req.session_id,
            project_id=req.project_id or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
            session_id=req.session_id,
            project_id=req.project_id or None,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422 if isinstance(exc, ValueError) else 404, detail=str(exc)) from exc
    return MemoryWriteResponse(memory_id=updated.memory_id, version=updated.version)


@app.post("/retrieve_store", response_model=RetrievalResponse)
async def retrieve_store(req: RetrievalRequest):
    """Legacy endpoint - only available in experimental_global mode."""
    if memory_mode == "session":
        raise HTTPException(
            status_code=409,
            detail="retrieve_store endpoint is disabled in session mode; use /context/recent instead"
        )
    
    import time
    start = time.time()
    outcome = memory_service.retrieve_store(
        req.query,
        scope_level=req.scope_level,
        destination=req.destination,
        top_k=req.top_k,
        session_id=req.session_id,
        project_id=req.project_id or None,
    )
    assembled = outcome["assembled"]
    retrieval_event_id = record_retrieval(
        shadow_collector,
        enabled=shadow_enabled and req.session_id is not None and req.turn_id is not None,
        session_id=req.session_id or "",
        turn_id=req.turn_id or "",
        query={"text": req.query, "scope": {"level": req.scope_level}, "destination": req.destination},
        candidates=outcome["ranked"],
        metadata={
            "ranker_model_id": memory_service.pipeline.ranker.model_id,
            "injected_count": len(assembled),
            "route_path": outcome["decision"].primary_path,
        },
    )
    return RetrievalResponse(
        results=assembled,
        latency_ms=round((time.time() - start) * 1000, 2),
        retrieval_event_id=retrieval_event_id,
    )


@app.post("/conflicts")
async def conflicts(req: RetrievalRequest):
    """Legacy endpoint - only available in experimental_global mode."""
    if memory_mode == "session":
        raise HTTPException(
            status_code=409,
            detail="conflicts endpoint is disabled in session mode"
        )
    
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
    """Legacy endpoint - only available in experimental_global mode."""
    if memory_mode == "session":
        raise HTTPException(
            status_code=409,
            detail="route endpoint is disabled in session mode"
        )
    
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
    """Return write suggestions; this endpoint never commits them.
    
    Only available in experimental_global mode.
    """
    if memory_mode != "experimental_global":
        raise HTTPException(
            status_code=409,
            detail="write_proposals endpoint is only available in experimental_global mode"
        )
    return {"proposals": proposal_payload(req.text, scope_level=req.scope_level)}


@app.post("/context/turn", response_model=ContextTurnResponse)
async def record_context_turn(req: ContextTurnRequest):
    # Server-side payload filtering
    is_valid, error_msg = filter_turn_payload(req.user_text, req.assistant_text)
    if not is_valid:
        raise HTTPException(status_code=422, detail=error_msg)
    
    try:
        memory_id = context_service.record_turn(
            session_id=req.session_id,
            turn_id=req.turn_id,
            project_id=req.project_id,
            user_text=req.user_text,
            assistant_text=req.assistant_text,
            destination=req.destination,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ContextTurnResponse(memory_id=memory_id)


@app.post("/context/checkpoint", response_model=ContextCheckpointResponse)
async def record_context_checkpoint(req: ContextCheckpointRequest):
    memory_id = context_service.record_checkpoint(
        session_id=req.session_id,
        turn_id=req.turn_id,
        project_id=req.project_id,
        transcript=req.transcript,
        destination=req.destination,
    )
    return ContextCheckpointResponse(memory_id=memory_id)


class ContextRecentRequest(BaseModel):
    session_id: str
    project_id: str = ""
    max_tokens: int = Field(default=4000, ge=100, le=16000)
    tokens_per_memory: int = Field(default=200, ge=50, le=1000)


class ContextRecentResponse(BaseModel):
    results: List[dict]
    token_cost: int


@app.post("/context/recent", response_model=ContextRecentResponse)
async def retrieve_recent_context(req: ContextRecentRequest):
    """Retrieve recent conversation context for the session in chronological order."""
    if not req.session_id:
        raise HTTPException(status_code=422, detail="session_id is required")
    
    try:
        outcome = context_service.retrieve_recent(
            session_id=req.session_id,
            project_id=req.project_id,
            max_tokens=req.max_tokens,
            tokens_per_memory=req.tokens_per_memory,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    
    return ContextRecentResponse(
        results=outcome["results"],
        token_cost=outcome["token_cost"],
    )


@app.post("/shadow/citations")
async def log_citation(req: CitationLogRequest):
    """Legacy endpoint - only available in experimental_global mode with MNEMOSYNE_EXPERIMENTAL_SHADOW=1."""
    if memory_mode == "session":
        raise HTTPException(
            status_code=409,
            detail="shadow/citations endpoint is disabled in session mode; set MNEMOSYNE_MEMORY_MODE=experimental_global and MNEMOSYNE_EXPERIMENTAL_SHADOW=1"
        )
    
    event_id = record_citation(
        shadow_collector,
        enabled=shadow_enabled,
        session_id=req.session_id,
        turn_id=req.turn_id,
        retrieval_event_id=req.retrieval_event_id,
        memory_id=req.memory_id,
        citation=req.citation,
        original_rank=req.original_rank,
        context_position=req.context_position,
    )
    if event_id is None:
        raise HTTPException(status_code=409, detail="shadow mode is disabled")
    return {"event_id": event_id}


@app.post("/shadow/corrections")
async def log_correction(req: CorrectionLogRequest):
    """Legacy endpoint - only available in experimental_global mode with MNEMOSYNE_EXPERIMENTAL_SHADOW=1."""
    if memory_mode == "session":
        raise HTTPException(
            status_code=409,
            detail="shadow/corrections endpoint is disabled in session mode; set MNEMOSYNE_MEMORY_MODE=experimental_global and MNEMOSYNE_EXPERIMENTAL_SHADOW=1"
        )
    
    event_id = record_correction(
        shadow_collector,
        enabled=shadow_enabled,
        session_id=req.session_id,
        turn_id=req.turn_id,
        citation_event_id=req.citation_event_id,
        memory_id=req.memory_id,
        correction_type=req.correction_type,
        feedback=req.feedback,
    )
    if event_id is None:
        raise HTTPException(status_code=409, detail="shadow mode is disabled")
    return {"event_id": event_id}


@app.get("/shadow/metrics")
def shadow_metrics(session_id: str | None = None):
    """Legacy endpoint - only available in experimental_global mode with MNEMOSYNE_EXPERIMENTAL_SHADOW=1."""
    if memory_mode == "session":
        raise HTTPException(
            status_code=409,
            detail="shadow/metrics endpoint is disabled in session mode; set MNEMOSYNE_MEMORY_MODE=experimental_global and MNEMOSYNE_EXPERIMENTAL_SHADOW=1"
        )
    
    if not shadow_enabled or shadow_collector is None:
        raise HTTPException(status_code=409, detail="shadow mode is disabled")
    return shadow_collector.compute_metrics(session_id)


@app.post("/detect_citations", response_model=CitationResponse)
async def detect_citations(req: CitationRequest):
    """
    Detect which memories were cited in a model response.
    
    Legacy endpoint - only available in experimental_global mode.
    """
    if memory_mode == "session":
        raise HTTPException(
            status_code=409,
            detail="detect_citations endpoint is disabled in session mode"
        )
    
    try:
        injected = [
            {
                "memory_id": m.resolved_id,
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
    """List available ranking models.
    
    Legacy endpoint - only available in experimental_global mode.
    """
    if memory_mode == "session":
        raise HTTPException(
            status_code=409,
            detail="models endpoint is disabled in session mode"
        )
    
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


class ExperimentalGlobalMemoryRequest(BaseModel):
    id: str
    content: str
    approved: bool
    source_session_id: str = ""


class ExperimentalGlobalMemoryResponse(BaseModel):
    memory_id: str
    version: int
    created_at: float
    updated_at: float
    approved: bool


class ExperimentalGlobalRetrieveRequest(BaseModel):
    query: str
    max_tokens: int = Field(default=1000, ge=100, le=5000)
    tokens_per_memory: int = Field(default=200, ge=50, le=500)


class ExperimentalGlobalRetrieveResponse(BaseModel):
    results: List[dict]
    retrieval_event_id: str | None = None


@app.post("/experimental/global-memories", response_model=ExperimentalGlobalMemoryResponse)
async def write_experimental_global_memory(req: ExperimentalGlobalMemoryRequest):
    """Write an experimental cross-conversation global memory.
    
    Only available when MNEMOSYNE_MEMORY_MODE=experimental_global.
    Requires approved=true; never auto-commits proposals.
    """
    if experimental_global is None:
        raise HTTPException(
            status_code=409,
            detail="experimental global memory is disabled; set MNEMOSYNE_MEMORY_MODE=experimental_global"
        )
    
    try:
        result = experimental_global.write(
            memory_id=req.id,
            content=req.content,
            approved=req.approved,
            source_session_id=req.source_session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    
    return ExperimentalGlobalMemoryResponse(**result)


@app.get("/experimental/global-memories")
async def list_experimental_global_memories(limit: int = 100):
    """List all experimental global memories.
    
    Only available when MNEMOSYNE_MEMORY_MODE=experimental_global.
    """
    if experimental_global is None:
        raise HTTPException(
            status_code=409,
            detail="experimental global memory is disabled; set MNEMOSYNE_MEMORY_MODE=experimental_global"
        )
    
    return {"memories": experimental_global.list_all(limit=limit)}


@app.post("/experimental/global-memories/retrieve", response_model=ExperimentalGlobalRetrieveResponse)
async def retrieve_experimental_global_memories(req: ExperimentalGlobalRetrieveRequest):
    """Retrieve approved experimental global memories relevant to a query.
    
    Only available when MNEMOSYNE_MEMORY_MODE=experimental_global.
    """
    if experimental_global is None:
        raise HTTPException(
            status_code=409,
            detail="experimental global memory is disabled; set MNEMOSYNE_MEMORY_MODE=experimental_global"
        )
    
    results = experimental_global.retrieve(
        query=req.query,
        max_tokens=req.max_tokens,
        tokens_per_memory=req.tokens_per_memory,
    )
    
    return ExperimentalGlobalRetrieveResponse(results=results, retrieval_event_id=None)


@app.get("/")
def root():
    """Root endpoint with API info.
    
    Endpoints listed are mode-aware:
    - session mode: only session context endpoints
    - experimental_global mode: session context + experimental endpoints
    """
    base_endpoints = {
        "health": "/health",
        "context_turn": "/context/turn",
        "context_recent": "/context/recent",
    }
    
    # Optionally include checkpoint if still needed
    if memory_mode == "session":
        base_endpoints["context_checkpoint"] = "/context/checkpoint"
    
    if memory_mode == "experimental_global":
        # Add experimental endpoints
        base_endpoints.update({
            "context_checkpoint": "/context/checkpoint",
            "write_proposals": "/write_proposals",
            "experimental_global_retrieve": "/experimental/global-memories/retrieve",
            "experimental_global_list": "/experimental/global-memories",
        })
    
    return {
        "service": "MNEMOSYNE-MNS",
        "version": "1.0.0",
        "memory_mode": memory_mode,
        "endpoints": base_endpoints,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
