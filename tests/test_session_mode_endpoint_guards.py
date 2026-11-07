"""HTTP endpoint 409 contract tests for session mode.

Validates that legacy endpoints return explicit 409 feature-disabled
responses in session mode instead of just being hidden from discovery.

This is a TOY test (no real data) focusing on the contract:
- Covered endpoints should return 409 in session mode
- Deterministic sorting in experimental_global_memory

Note: Full HTTP integration tests require PyTorch/model loading and are
skipped in the local environment. These tests validate the code paths exist.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add scripts to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TestEndpointGuardLogic(unittest.TestCase):
    """Test that endpoint guard logic is correctly implemented.
    
    These are unit tests of the guard logic, not full HTTP integration tests.
    """
    
    def test_session_mode_guard_pattern_exists(self):
        """Verify that session mode guards are present in server.py."""
        server_path = PROJECT_ROOT / "docker" / "server.py"
        content = server_path.read_text(encoding="utf-8")
        
        # Check that guards exist for key endpoints
        required_guards = [
            ('"/retrieve"', 'memory_mode == "session"'),
            ('"/retrieve_store"', 'memory_mode == "session"'),
            ('"/route"', 'memory_mode == "session"'),
            ('"/conflicts"', 'memory_mode == "session"'),
            ('"/shadow/citations"', 'memory_mode == "session"'),
            ('"/shadow/corrections"', 'memory_mode == "session"'),
            ('"/shadow/metrics"', 'memory_mode == "session"'),
            ('"/detect_citations"', 'memory_mode == "session"'),
            ('"/models"', 'memory_mode == "session"'),
        ]
        
        for endpoint, guard_condition in required_guards:
            self.assertIn(endpoint, content, f"Endpoint {endpoint} should be defined")
            # Find the endpoint definition
            endpoint_start = content.find(endpoint)
            # Look for the guard within next 1000 chars
            endpoint_section = content[endpoint_start:endpoint_start + 1000]
            self.assertIn(
                guard_condition,
                endpoint_section,
                f"Endpoint {endpoint} should have session mode guard"
            )
            self.assertIn(
                "HTTPException",
                endpoint_section,
                f"Endpoint {endpoint} should raise HTTPException"
            )
            self.assertIn(
                "409",
                endpoint_section,
                f"Endpoint {endpoint} should return 409"
            )
    
    def test_health_endpoint_has_no_mode_guard(self):
        """Verify /health endpoint does not have mode-based guards."""
        server_path = PROJECT_ROOT / "docker" / "server.py"
        content = server_path.read_text(encoding="utf-8")
        
        # Find health endpoint
        health_start = content.find('"/health"')
        self.assertNotEqual(health_start, -1, "/health endpoint should exist")
        
        # Check next 500 chars after /health definition
        health_section = content[health_start:health_start + 500]
        
        # Should NOT contain mode guard before return
        return_pos = health_section.find("return {")
        if return_pos != -1:
            before_return = health_section[:return_pos]
            self.assertNotIn(
                'memory_mode == "session"',
                before_return,
                "/health should not check memory_mode"
            )


class TestDeterministicSorting(unittest.TestCase):
    """Test deterministic sorting in experimental_global_memory.py."""
    
    def test_same_timestamp_deterministic_order(self):
        """Entries with same created_at should have deterministic order via memory_id."""
        from experimental_global_memory import ExperimentalGlobalMemory
        from memory_store import MemoryStore
        import tempfile
        import sqlite3
        
        # Create temp db
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = MemoryStore(db_path)
            exp_global = ExperimentalGlobalMemory(store)
            
            # Insert 3 entries with same timestamp
            now = 1234567890.0
            conn = sqlite3.connect(db_path)
            conn.execute("""
                INSERT INTO experimental_global_memories 
                (memory_id, content, approved, created_at, updated_at, status)
                VALUES 
                ('experimental_global:z', 'Content Z', 1, ?, ?, 'active'),
                ('experimental_global:a', 'Content A', 1, ?, ?, 'active'),
                ('experimental_global:m', 'Content M', 1, ?, ?, 'active')
            """, (now, now, now, now, now, now))
            conn.commit()
            conn.close()
            
            # Retrieve twice - should be deterministic
            result1 = exp_global.list_all(limit=10)
            result2 = exp_global.list_all(limit=10)
            
            # Extract IDs
            ids1 = [r["memory_id"] for r in result1]
            ids2 = [r["memory_id"] for r in result2]
            
            # Should be identical and in alphabetical order by ID (since timestamp is same)
            self.assertEqual(ids1, ids2, "Results should be deterministic")
            self.assertEqual(
                ids1,
                ["experimental_global:a", "experimental_global:m", "experimental_global:z"],
                "Should be sorted by memory_id ASC when created_at is same"
            )
        
        finally:
            # Cleanup - close all connections first on Windows
            import gc
            gc.collect()  # Force garbage collection to close any lingering connections
            try:
                import os
                if os.path.exists(db_path):
                    os.unlink(db_path)
            except PermissionError:
                # Windows file lock - not critical for test validity
                pass
    
    def test_sql_order_by_clause_includes_memory_id(self):
        """Verify that SQL queries include memory_id in ORDER BY for determinism."""
        exp_global_path = PROJECT_ROOT / "scripts" / "experimental_global_memory.py"
        content = exp_global_path.read_text(encoding="utf-8")
        
        # Check list_all() method
        list_all_start = content.find("def list_all(")
        self.assertNotEqual(list_all_start, -1, "list_all method should exist")
        list_all_section = content[list_all_start:list_all_start + 1500]
        
        self.assertIn(
            "ORDER BY created_at DESC, memory_id ASC",
            list_all_section,
            "list_all should order by created_at DESC, memory_id ASC"
        )
        
        # Check retrieve() method
        retrieve_start = content.find("def retrieve(")
        self.assertNotEqual(retrieve_start, -1, "retrieve method should exist")
        retrieve_section = content[retrieve_start:retrieve_start + 2000]
        
        self.assertIn(
            "ORDER BY created_at DESC, memory_id ASC",
            retrieve_section,
            "retrieve should order by created_at DESC, memory_id ASC"
        )


class TestDocumentationClarity(unittest.TestCase):
    """Test that documentation clearly separates session vs experimental modes."""
    
    def test_hermes_integration_doc_has_mode_sections(self):
        """Verify HERMES-INTEGRATION.md has clear session vs experimental sections."""
        doc_path = PROJECT_ROOT / "docs" / "HERMES-INTEGRATION.md"
        content = doc_path.read_text(encoding="utf-8")
        
        # Should have both mode sections
        self.assertIn("### Session Mode (Default)", content)
        self.assertIn("### Experimental Global Mode", content)
        
        # Should have lifecycle sections for both
        self.assertIn("## Lifecycle", content)
        
        # Check that legacy path is marked
        self.assertIn("legacy", content.lower(), "Should mark legacy/experimental paths")
    
    def test_legacy_lifecycle_annotated(self):
        """Verify that ROUTE→RANK→ASSEMBLE lifecycle is marked as legacy."""
        doc_path = PROJECT_ROOT / "docs" / "HERMES-INTEGRATION.md"
        content = doc_path.read_text(encoding="utf-8")
        
        # Find ROUTE section
        route_pos = content.find("ROUTE")
        self.assertNotEqual(route_pos, -1, "ROUTE lifecycle should be documented")
        
        # Check surrounding context (within 500 chars before/after)
        context_start = max(0, route_pos - 500)
        context_end = min(len(content), route_pos + 500)
        context = content[context_start:context_end]
        
        # Should be marked as experimental or legacy
        has_marker = any(marker in context.lower() for marker in ["legacy", "experimental"])
        self.assertTrue(
            has_marker,
            "ROUTE lifecycle should be marked as legacy or experimental"
        )


if __name__ == "__main__":
    unittest.main()
