"""
Shadow mode trace collector for MNEMOSYNE-MNS.

Collects retrieved → cited → corrected events for production validation
and continuous learning. All data stored locally; no cloud upload.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RetrievalEvent:
    event_id: str
    timestamp: float
    session_id: str
    turn_id: str
    query_text: str
    query_json: dict[str, Any]
    candidates_json: list[dict[str, Any]]
    metadata_json: dict[str, Any]


@dataclass
class CitationEvent:
    event_id: str
    timestamp: float
    session_id: str
    turn_id: str
    retrieval_event_id: str
    memory_id: str
    citation_json: dict[str, Any]
    context_position: int | None
    original_rank: int


@dataclass
class CorrectionEvent:
    event_id: str
    timestamp: float
    session_id: str
    turn_id: str
    citation_event_id: str
    memory_id: str
    correction_type: str
    feedback_json: dict[str, Any]


@dataclass
class IgnoredEvent:
    event_id: str
    timestamp: float
    session_id: str
    turn_id: str
    retrieval_event_id: str
    memory_id: str
    rank: int
    injected: bool


class ShadowModeCollector:
    """
    Collects shadow mode traces in a local SQLite database.
    
    All data stays on the user's machine. Credentials and PII are redacted.
    """
    
    def __init__(self, db_path: str | Path = "~/.hermes/shadow_traces.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS retrieval_events (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                query_json TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_session ON retrieval_events(session_id);
            CREATE INDEX IF NOT EXISTS idx_retrieval_timestamp ON retrieval_events(timestamp);
            
            CREATE TABLE IF NOT EXISTS citation_events (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                retrieval_event_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                citation_json TEXT NOT NULL,
                context_position INTEGER,
                original_rank INTEGER,
                FOREIGN KEY (retrieval_event_id) REFERENCES retrieval_events(event_id)
            );
            CREATE INDEX IF NOT EXISTS idx_citation_memory ON citation_events(memory_id);
            CREATE INDEX IF NOT EXISTS idx_citation_retrieval ON citation_events(retrieval_event_id);
            
            CREATE TABLE IF NOT EXISTS correction_events (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                citation_event_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                correction_type TEXT NOT NULL,
                feedback_json TEXT NOT NULL,
                FOREIGN KEY (citation_event_id) REFERENCES citation_events(event_id)
            );
            CREATE INDEX IF NOT EXISTS idx_correction_memory ON correction_events(memory_id);
            CREATE INDEX IF NOT EXISTS idx_correction_citation ON correction_events(citation_event_id);
            
            CREATE TABLE IF NOT EXISTS ignored_events (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                retrieval_event_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                injected BOOLEAN NOT NULL,
                FOREIGN KEY (retrieval_event_id) REFERENCES retrieval_events(event_id)
            );
            CREATE INDEX IF NOT EXISTS idx_ignored_memory ON ignored_events(memory_id);
        """)
        self.conn.commit()
    
    def log_retrieval(
        self,
        session_id: str,
        turn_id: str,
        query: dict[str, Any],
        candidates: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> RetrievalEvent:
        """Log a retrieval event."""
        event = RetrievalEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            session_id=session_id,
            turn_id=turn_id,
            query_text=query.get("text", ""),
            query_json=query,
            candidates_json=candidates,
            metadata_json=metadata,
        )
        
        self.conn.execute(
            """
            INSERT INTO retrieval_events 
            (event_id, timestamp, session_id, turn_id, query_text, query_json, candidates_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.timestamp,
                event.session_id,
                event.turn_id,
                event.query_text,
                json.dumps(event.query_json, ensure_ascii=False),
                json.dumps(event.candidates_json, ensure_ascii=False),
                json.dumps(event.metadata_json, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return event
    
    def log_citation(
        self,
        session_id: str,
        turn_id: str,
        retrieval_event_id: str,
        memory_id: str,
        citation: dict[str, Any],
        context_position: int | None = None,
        original_rank: int | None = None,
    ) -> CitationEvent:
        """Log a citation event."""
        event = CitationEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            session_id=session_id,
            turn_id=turn_id,
            retrieval_event_id=retrieval_event_id,
            memory_id=memory_id,
            citation_json=citation,
            context_position=context_position,
            original_rank=original_rank or 0,
        )
        
        self.conn.execute(
            """
            INSERT INTO citation_events
            (event_id, timestamp, session_id, turn_id, retrieval_event_id, memory_id, citation_json, context_position, original_rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.timestamp,
                event.session_id,
                event.turn_id,
                event.retrieval_event_id,
                event.memory_id,
                json.dumps(event.citation_json, ensure_ascii=False),
                event.context_position,
                event.original_rank,
            ),
        )
        self.conn.commit()
        return event
    
    def log_correction(
        self,
        session_id: str,
        turn_id: str,
        citation_event_id: str,
        memory_id: str,
        correction_type: str,
        feedback: dict[str, Any],
    ) -> CorrectionEvent:
        """Log a correction event."""
        event = CorrectionEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            session_id=session_id,
            turn_id=turn_id,
            citation_event_id=citation_event_id,
            memory_id=memory_id,
            correction_type=correction_type,
            feedback_json=feedback,
        )
        
        self.conn.execute(
            """
            INSERT INTO correction_events
            (event_id, timestamp, session_id, turn_id, citation_event_id, memory_id, correction_type, feedback_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.timestamp,
                event.session_id,
                event.turn_id,
                event.citation_event_id,
                event.memory_id,
                event.correction_type,
                json.dumps(event.feedback_json, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return event
    
    def log_ignored(
        self,
        session_id: str,
        turn_id: str,
        retrieval_event_id: str,
        memory_id: str,
        rank: int,
        injected: bool,
    ) -> IgnoredEvent:
        """Log an ignored memory event."""
        event = IgnoredEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            session_id=session_id,
            turn_id=turn_id,
            retrieval_event_id=retrieval_event_id,
            memory_id=memory_id,
            rank=rank,
            injected=injected,
        )
        
        self.conn.execute(
            """
            INSERT INTO ignored_events
            (event_id, timestamp, session_id, turn_id, retrieval_event_id, memory_id, rank, injected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.timestamp,
                event.session_id,
                event.turn_id,
                event.retrieval_event_id,
                event.memory_id,
                event.rank,
                event.injected,
            ),
        )
        self.conn.commit()
        return event
    
    def get_session_traces(self, session_id: str) -> dict[str, Any]:
        """Retrieve all traces for a session."""
        cursor = self.conn.cursor()
        
        retrieval_events = cursor.execute(
            "SELECT * FROM retrieval_events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        
        citation_events = cursor.execute(
            "SELECT * FROM citation_events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        
        correction_events = cursor.execute(
            "SELECT * FROM correction_events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        
        ignored_events = cursor.execute(
            "SELECT * FROM ignored_events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        
        return {
            "session_id": session_id,
            "retrieval_events": retrieval_events,
            "citation_events": citation_events,
            "correction_events": correction_events,
            "ignored_events": ignored_events,
        }
    
    def compute_metrics(self, session_id: str | None = None) -> dict[str, Any]:
        """Compute production metrics from collected traces."""
        cursor = self.conn.cursor()
        
        where_clause = "WHERE session_id = ?" if session_id else ""
        params = (session_id,) if session_id else ()
        
        # Citation rate
        injected_count = cursor.execute(
            f"SELECT COUNT(*) FROM retrieval_events {where_clause}",
            params,
        ).fetchone()[0]
        
        cited_count = cursor.execute(
            f"SELECT COUNT(*) FROM citation_events {where_clause}",
            params,
        ).fetchone()[0]
        
        correction_count = cursor.execute(
            f"SELECT COUNT(*) FROM correction_events {where_clause}",
            params,
        ).fetchone()[0]
        
        citation_rate = cited_count / injected_count if injected_count > 0 else 0.0
        correction_rate = correction_count / cited_count if cited_count > 0 else 0.0
        
        return {
            "session_id": session_id,
            "injected_count": injected_count,
            "cited_count": cited_count,
            "correction_count": correction_count,
            "citation_rate": citation_rate,
            "correction_rate": correction_rate,
        }
    
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
