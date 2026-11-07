"""Tests for shadow trace training data export pipeline.

All tests use temporary toy fixtures. No real user databases are accessed.
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class ShadowTrainingExportTest(unittest.TestCase):
    """Test shadow trace export with toy fixtures only."""

    def test_export_requires_explicit_source_path(self):
        """Export must reject when no explicit source path is provided."""
        from build_shadow_training_data import export_shadow_traces
        
        with tempfile.TemporaryDirectory() as tmp_out:
            output_dir = Path(tmp_out)
            
            # No source provided - should fail with clear error
            with self.assertRaises(ValueError) as cm:
                export_shadow_traces(
                    source_db_path=None,
                    output_dir=output_dir,
                    include_text=False,
                )
            
            self.assertIn("explicit", str(cm.exception).lower())
            self.assertIn("source", str(cm.exception).lower())


    def test_export_opens_source_db_readonly(self):
        """Export must open source DB in read-only mode and not modify it."""
        from build_shadow_training_data import export_shadow_traces
        
        with tempfile.TemporaryDirectory() as tmp:
            # Create a toy shadow DB
            db_path = Path(tmp) / "shadow.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("""
                CREATE TABLE retrieval_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    query_json TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()
            
            # Record original mtime and hash
            original_mtime = db_path.stat().st_mtime
            original_size = db_path.stat().st_size
            
            with tempfile.TemporaryDirectory() as tmp_out:
                output_dir = Path(tmp_out)
                
                # Export should not modify source
                export_shadow_traces(
                    source_db_path=db_path,
                    output_dir=output_dir,
                    include_text=False,
                )
                
                # Verify source unchanged
                self.assertEqual(db_path.stat().st_mtime, original_mtime)
                self.assertEqual(db_path.stat().st_size, original_size)

    def test_cited_memory_generates_positive_label(self):
        """Cited memories become positive training samples."""
        from build_shadow_training_data import export_shadow_traces
        
        with tempfile.TemporaryDirectory() as tmp:
            # Create toy shadow DB with citation evidence
            db_path = Path(tmp) / "shadow.db"
            self._create_toy_shadow_db_with_citation(db_path)
            
            with tempfile.TemporaryDirectory() as tmp_out:
                output_dir = Path(tmp_out)
                
                result = export_shadow_traces(
                    source_db_path=db_path,
                    output_dir=output_dir,
                    include_text=False,
                )
                
                # Read generated samples
                samples_path = output_dir / "shadow_training_samples.jsonl"
                self.assertTrue(samples_path.exists())
                
                samples = []
                with open(samples_path) as f:
                    for line in f:
                        samples.append(json.loads(line))
                
                # Must have exactly 1 positive sample
                positives = [s for s in samples if s["label"] == "positive"]
                self.assertEqual(len(positives), 1)
                
                sample = positives[0]
                self.assertEqual(sample["memory_id"], "m1")
                self.assertEqual(sample["label_source"], "citation")
                self.assertEqual(sample["session_id"], "s1")
                self.assertIn("retrieval_event_id", sample)
                self.assertIn("citation_event_id", sample)
    
    def test_unretrieved_memory_generates_unknown_not_negative(self):
        """Memories not in retrieval results must be labeled unknown, not negative."""
        from build_shadow_training_data import export_shadow_traces
        
        with tempfile.TemporaryDirectory() as tmp:
            # Create toy DB with retrieval but no citation
            db_path = Path(tmp) / "shadow.db"
            conn = sqlite3.connect(str(db_path))
            conn.executescript("""
                CREATE TABLE retrieval_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    query_json TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
            """)
            
            # Insert retrieval with injected candidate that was NOT cited
            conn.execute(
                """INSERT INTO retrieval_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "r1",
                    1000.0,
                    "s1",
                    "t1",
                    "test query",
                    json.dumps({"text": "test query"}),
                    json.dumps([{"memory_id": "m1", "rank": 1, "score": 0.9, "injected": True}]),
                    json.dumps({"ranker_model_id": "TEST"}),
                ),
            )
            conn.commit()
            conn.close()
            
            with tempfile.TemporaryDirectory() as tmp_out:
                output_dir = Path(tmp_out)
                
                result = export_shadow_traces(
                    source_db_path=db_path,
                    output_dir=output_dir,
                    include_text=False,
                )
                
                # Read samples
                samples_path = output_dir / "shadow_training_samples.jsonl"
                samples = []
                with open(samples_path) as f:
                    for line in f:
                        samples.append(json.loads(line))
                
                # Should have NO negative labels
                negatives = [s for s in samples if s.get("label") == "negative"]
                self.assertEqual(len(negatives), 0)
                
                # P2 update: Uncited candidates now generate unknown samples (not omitted)
                self.assertEqual(len(samples), 1)
                unknown = [s for s in samples if s.get("label") == "unknown"]
                self.assertEqual(len(unknown), 1)
                self.assertIn("unknown_reason", unknown[0])

    def test_correction_event_generates_labeled_sample(self):
        """Correction events become labeled samples with correction evidence."""
        from build_shadow_training_data import export_shadow_traces
        
        with tempfile.TemporaryDirectory() as tmp:
            # Create toy DB with retrieval + citation + correction
            db_path = Path(tmp) / "shadow.db"
            conn = sqlite3.connect(str(db_path))
            conn.executescript("""
                CREATE TABLE retrieval_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    query_json TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE citation_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    retrieval_event_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    citation_json TEXT NOT NULL,
                    context_position INTEGER,
                    original_rank INTEGER
                );
                CREATE TABLE correction_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    citation_event_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    correction_type TEXT NOT NULL,
                    feedback_json TEXT NOT NULL
                );
            """)
            
            # Insert retrieval
            conn.execute(
                """INSERT INTO retrieval_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("r1", 1000.0, "s1", "t1", "query", json.dumps({"text": "query"}),
                 json.dumps([{"memory_id": "m1", "rank": 1, "score": 0.9, "injected": True}]),
                 json.dumps({"ranker_model_id": "TEST"})),
            )
            
            # Insert citation
            conn.execute(
                """INSERT INTO citation_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("c1", 1001.0, "s1", "t1", "r1", "m1", json.dumps({"span": "cited"}), 1, 1),
            )
            
            # Insert correction
            conn.execute(
                """INSERT INTO correction_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("corr1", 1002.0, "s1", "t2", "c1", "m1", "outdated", json.dumps({"text": "wrong"})),
            )
            
            conn.commit()
            conn.close()
            
            with tempfile.TemporaryDirectory() as tmp_out:
                output_dir = Path(tmp_out)
                
                export_shadow_traces(
                    source_db_path=db_path,
                    output_dir=output_dir,
                    include_text=False,
                )
                
                samples_path = output_dir / "shadow_training_samples.jsonl"
                samples = []
                with open(samples_path) as f:
                    for line in f:
                        samples.append(json.loads(line))
                
                # Should have samples with correction evidence
                corrections = [s for s in samples if s.get("correction_event_id") is not None]
                self.assertGreater(len(corrections), 0)
                
                corr_sample = corrections[0]
                self.assertEqual(corr_sample["correction_type"], "outdated")
                self.assertEqual(corr_sample["label_source"], "correction")

    def test_export_generates_manifest_without_leaking_paths(self):
        """Manifest must contain statistics but no absolute source paths."""
        from build_shadow_training_data import export_shadow_traces
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "shadow.db"
            self._create_toy_shadow_db_with_citation(db_path)
            
            with tempfile.TemporaryDirectory() as tmp_out:
                output_dir = Path(tmp_out)
                
                result = export_shadow_traces(
                    source_db_path=db_path,
                    output_dir=output_dir,
                    include_text=False,
                )
                
                # Manifest file should exist
                manifest_path = output_dir / "manifest.json"
                self.assertTrue(manifest_path.exists())
                
                with open(manifest_path) as f:
                    manifest = json.load(f)
                
                # Must have summary statistics
                self.assertIn("total_samples", manifest)
                self.assertIn("label_counts", manifest)
                self.assertIn("timestamp", manifest)
                
                # Must NOT contain absolute source path
                manifest_str = json.dumps(manifest)
                self.assertNotIn(str(db_path), manifest_str)
                self.assertNotIn("C:\\", manifest_str)
                self.assertNotIn("c:/", manifest_str.lower())
    
    def _create_toy_shadow_db_with_citation(self, db_path: Path) -> None:
        """Helper to create toy DB with retrieval + citation."""
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE retrieval_events (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                query_json TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE citation_events (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                retrieval_event_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                citation_json TEXT NOT NULL,
                context_position INTEGER,
                original_rank INTEGER
            );
        """)
        
        # Insert retrieval with candidate
        conn.execute(
            """INSERT INTO retrieval_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "r1",
                1000.0,
                "s1",
                "t1",
                "test query",
                json.dumps({"text": "test query"}),
                json.dumps([{"memory_id": "m1", "rank": 1, "score": 0.9, "injected": True}]),
                json.dumps({"ranker_model_id": "TEST"}),
            ),
        )
        
        # Insert citation
        conn.execute(
            """INSERT INTO citation_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "c1",
                1001.0,
                "s1",
                "t1",
                "r1",
                "m1",
                json.dumps({"span": "cited", "confidence": 0.9}),
                1,
                1,
            ),
        )
        
        conn.commit()
        conn.close()


if __name__ == "__main__":
    unittest.main()
