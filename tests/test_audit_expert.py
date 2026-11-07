"""
Tests for AUDIT Expert quality monitoring module.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from audit_expert import (
    Alert,
    AlertRule,
    AuditExpert,
    QualityMetrics,
    compare_with_baseline,
)


@pytest.fixture
def temp_db():
    """Create a temporary shadow traces database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create schema
    cursor.execute("""
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
    
    cursor.execute("""
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
        )
    """)
    
    cursor.execute("""
        CREATE TABLE correction_events (
            event_id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            citation_event_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            correction_type TEXT NOT NULL,
            feedback_json TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE ignored_events (
            event_id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            retrieval_event_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            injected BOOLEAN NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    db_path.unlink()


def populate_test_data(db_path: Path, scenario: str = "healthy"):
    """Populate database with test data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    now = time.time()
    
    if scenario == "healthy":
        # 10 retrievals, 7 with citations, 1 correction, low pollution
        for i in range(10):
            retrieval_id = f"ret_{i}"
            cursor.execute(
                "INSERT INTO retrieval_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    retrieval_id,
                    now - (9 - i) * 3600,  # Spread over time
                    "session_1",
                    f"turn_{i}",
                    f"query {i}",
                    json.dumps({"type": "normal"}),
                    json.dumps([f"mem_{j}" for j in range(5)]),
                    json.dumps({"latency_ms": 100 + i * 10}),
                ),
            )
            
            # 70% citation rate
            if i < 7:
                cursor.execute(
                    "INSERT INTO citation_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"cite_{i}",
                        now - (9 - i) * 3600,
                        "session_1",
                        f"turn_{i}",
                        retrieval_id,
                        f"mem_{i % 5}",
                        json.dumps({"context": "test"}),
                        i,
                        0,
                    ),
                )
            
            # Add some ignored events (5% pollution)
            if i % 20 == 0:
                cursor.execute(
                    "INSERT INTO ignored_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"ignore_{i}",
                        now - (9 - i) * 3600,
                        "session_1",
                        f"turn_{i}",
                        retrieval_id,
                        f"mem_injected_{i}",
                        4,
                        True,
                    ),
                )
        
        # 1 correction (1/7 = 14% correction rate)
        cursor.execute(
            "INSERT INTO correction_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "corr_1",
                now - 3600,
                "session_1",
                "turn_1",
                "cite_1",
                "mem_1",
                "factual_error",
                json.dumps({"feedback": "incorrect"}),
            ),
        )
    
    elif scenario == "critical":
        # High correction rate, high pollution
        for i in range(10):
            retrieval_id = f"ret_{i}"
            cursor.execute(
                "INSERT INTO retrieval_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    retrieval_id,
                    now - (9 - i) * 3600,
                    "session_1",
                    f"turn_{i}",
                    f"query {i}",
                    json.dumps({"type": "normal"}),
                    json.dumps([f"mem_{j}" for j in range(5)]),
                    json.dumps({"latency_ms": 700 + i * 10}),  # High latency
                ),
            )
            
            # All retrievals cited
            cursor.execute(
                "INSERT INTO citation_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"cite_{i}",
                    now - (9 - i) * 3600,
                    "session_1",
                    f"turn_{i}",
                    retrieval_id,
                    f"mem_{i % 5}",
                    json.dumps({"context": "test"}),
                    i,
                    0,
                ),
            )
            
            # High correction rate (20%)
            if i % 5 == 0:
                cursor.execute(
                    "INSERT INTO correction_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"corr_{i}",
                        now - (9 - i) * 3600,
                        "session_1",
                        f"turn_{i}",
                        f"cite_{i}",
                        f"mem_{i % 5}",
                        "factual_error",
                        json.dumps({"feedback": "incorrect"}),
                    ),
                )
            
            # High pollution (20%)
            if i % 5 == 0:
                cursor.execute(
                    "INSERT INTO ignored_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"ignore_{i}",
                        now - (9 - i) * 3600,
                        "session_1",
                        f"turn_{i}",
                        retrieval_id,
                        f"mem_injected_{i}",
                        4,
                        True,
                    ),
                )
    
    elif scenario == "empty":
        # No data
        pass
    
    conn.commit()
    conn.close()


class TestAuditExpert:
    """Test AUDIT Expert functionality."""
    
    def test_initialization(self, temp_db):
        """Test expert initialization."""
        expert = AuditExpert(db_path=temp_db)
        assert expert.db_path == temp_db
        assert len(expert.alert_rules) > 0
    
    def test_initialization_missing_db(self):
        """Test initialization with missing database."""
        with pytest.raises(FileNotFoundError):
            AuditExpert(db_path="nonexistent.db")
    
    def test_compute_metrics_empty_db(self, temp_db):
        """Test metrics computation with empty database."""
        populate_test_data(temp_db, scenario="empty")
        expert = AuditExpert(db_path=temp_db)
        
        metrics = expert.compute_metrics(window_days=7)
        
        assert metrics.citation_rate == 0.0
        assert metrics.correction_rate == 0.0
        assert metrics.context_pollution == 0.0
        assert metrics.retrieval_latency_p99 == 0.0
        assert metrics.memory_growth_rate == 0.0
        assert metrics.unknown_ratio == 0.0
        assert metrics.total_retrievals == 0
    
    def test_compute_metrics_healthy(self, temp_db):
        """Test metrics computation with healthy data."""
        populate_test_data(temp_db, scenario="healthy")
        expert = AuditExpert(db_path=temp_db)
        
        metrics = expert.compute_metrics(window_days=7)
        
        # 7/10 retrievals have citations
        assert metrics.citation_rate == 0.7
        # 1/7 citations corrected
        assert metrics.correction_rate == pytest.approx(1/7, rel=0.01)
        # Low pollution (1 injected / 50 total candidates)
        assert metrics.context_pollution == pytest.approx(0.02, rel=0.01)
        # Latency should be around 100-190ms, P99 close to max
        assert 150 < metrics.retrieval_latency_p99 < 200
        # Memory growth rate
        assert metrics.memory_growth_rate > 0
        assert metrics.total_retrievals == 10
        assert metrics.total_citations == 7
        assert metrics.total_corrections == 1
    
    def test_compute_metrics_critical(self, temp_db):
        """Test metrics computation with critical data."""
        populate_test_data(temp_db, scenario="critical")
        expert = AuditExpert(db_path=temp_db)
        
        metrics = expert.compute_metrics(window_days=7)
        
        # All retrievals cited
        assert metrics.citation_rate == 1.0
        # 2/10 citations corrected = 20%
        assert metrics.correction_rate == 0.2
        # High pollution (2 injected / 50 candidates = 4%)
        assert metrics.context_pollution == pytest.approx(0.04, rel=0.01)
        # High latency (700+)
        assert metrics.retrieval_latency_p99 > 700
    
    def test_check_alerts_healthy(self, temp_db):
        """Test alert checking with healthy metrics."""
        populate_test_data(temp_db, scenario="healthy")
        expert = AuditExpert(db_path=temp_db)
        
        metrics = expert.compute_metrics(window_days=7)
        alerts = expert.check_alerts(metrics)
        
        # Should have no critical alerts
        critical_alerts = [a for a in alerts if a.rule.severity == "critical"]
        assert len(critical_alerts) == 0
    
    def test_check_alerts_critical(self, temp_db):
        """Test alert checking with critical metrics."""
        populate_test_data(temp_db, scenario="critical")
        expert = AuditExpert(db_path=temp_db)
        
        metrics = expert.compute_metrics(window_days=7)
        alerts = expert.check_alerts(metrics)
        
        # Should have alerts
        assert len(alerts) > 0
        
        # Should have critical latency alert
        latency_alerts = [a for a in alerts if a.rule.metric_name == "retrieval_latency_p99"]
        assert len(latency_alerts) > 0
        
        # Should have critical correction rate alert
        correction_alerts = [a for a in alerts if a.rule.metric_name == "correction_rate"]
        assert len(correction_alerts) > 0
    
    def test_custom_alert_rules(self, temp_db):
        """Test custom alert rules."""
        custom_rules = [
            AlertRule(
                metric_name="citation_rate",
                operator="<",
                threshold=0.5,
                severity="warning",
                message="Citation rate below 50%",
            ),
        ]
        
        populate_test_data(temp_db, scenario="healthy")
        expert = AuditExpert(db_path=temp_db, alert_rules=custom_rules)
        
        metrics = expert.compute_metrics(window_days=7)
        alerts = expert.check_alerts(metrics)
        
        # Citation rate is 70%, so no alert
        assert len(alerts) == 0
    
    def test_generate_report(self, temp_db):
        """Test report generation."""
        populate_test_data(temp_db, scenario="healthy")
        expert = AuditExpert(db_path=temp_db)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            report = expert.generate_report(window_days=7, output_path=output_path)
            
            assert "generated_at" in report
            assert "window_days" in report
            assert "metrics" in report
            assert "alerts" in report
            assert "status" in report
            assert report["status"] in ["healthy", "warning", "critical"]
            
            # File should be written
            assert output_path.exists()
            loaded_report = json.loads(output_path.read_text())
            assert loaded_report == report
    
    def test_metrics_to_dict(self):
        """Test QualityMetrics serialization."""
        metrics = QualityMetrics(
            window_start=1000.0,
            window_end=2000.0,
            citation_rate=0.7,
            correction_rate=0.1,
            context_pollution=0.05,
            retrieval_latency_p99=150.0,
            memory_growth_rate=5.0,
            unknown_ratio=0.02,
            total_retrievals=100,
            total_citations=70,
            total_corrections=7,
            total_ignored_injected=5,
            total_candidates=1000,
        )
        
        data = metrics.to_dict()
        
        assert data["window_start"] == 1000.0
        assert data["window_end"] == 2000.0
        assert data["metrics"]["citation_rate"] == 0.7
        assert data["counts"]["total_retrievals"] == 100
    
    def test_alert_to_dict(self):
        """Test Alert serialization."""
        rule = AlertRule(
            metric_name="citation_rate",
            operator="<",
            threshold=0.5,
            severity="warning",
            message="Low citation rate",
        )
        alert = Alert(rule=rule, actual_value=0.3, timestamp=1000.0)
        
        data = alert.to_dict()
        
        assert data["metric"] == "citation_rate"
        assert data["severity"] == "warning"
        assert data["actual_value"] == 0.3
        assert data["threshold"] == 0.5


class TestBaselineComparison:
    """Test baseline comparison functionality."""
    
    def test_compare_with_baseline(self):
        """Test baseline comparison."""
        current = QualityMetrics(
            window_start=2000.0,
            window_end=3000.0,
            citation_rate=0.75,
            correction_rate=0.08,
            context_pollution=0.04,
            retrieval_latency_p99=120.0,
            memory_growth_rate=6.0,
            unknown_ratio=0.01,
            total_retrievals=100,
            total_citations=75,
            total_corrections=6,
            total_ignored_injected=4,
            total_candidates=1000,
        )
        
        baseline = QualityMetrics(
            window_start=1000.0,
            window_end=2000.0,
            citation_rate=0.7,
            correction_rate=0.1,
            context_pollution=0.05,
            retrieval_latency_p99=150.0,
            memory_growth_rate=5.0,
            unknown_ratio=0.02,
            total_retrievals=100,
            total_citations=70,
            total_corrections=10,
            total_ignored_injected=5,
            total_candidates=1000,
        )
        
        results = compare_with_baseline(current, baseline, n_resamples=100, seed=42)
        
        # Check structure
        assert "citation_rate" in results
        assert "correction_rate" in results
        assert "context_pollution" in results
        
        # Check citation rate improvement
        assert results["citation_rate"]["delta"] > 0
        assert results["citation_rate"]["current"] == 0.75
        assert results["citation_rate"]["baseline"] == 0.7
        
        # Check correction rate improvement (lower is better)
        assert results["correction_rate"]["delta"] < 0
        assert results["correction_rate"]["current"] == 0.08
        assert results["correction_rate"]["baseline"] == 0.1


class TestAlertRule:
    """Test AlertRule functionality."""
    
    def test_alert_rule_check_greater_equal(self):
        """Test >= operator."""
        rule = AlertRule(
            metric_name="citation_rate",
            operator=">=",
            threshold=0.5,
            severity="warning",
            message="Test",
        )
        
        assert rule.check(0.6) is True
        assert rule.check(0.5) is True
        assert rule.check(0.4) is False
    
    def test_alert_rule_check_less_equal(self):
        """Test <= operator."""
        rule = AlertRule(
            metric_name="correction_rate",
            operator="<=",
            threshold=0.15,
            severity="warning",
            message="Test",
        )
        
        assert rule.check(0.1) is True
        assert rule.check(0.15) is True
        assert rule.check(0.2) is False
    
    def test_alert_rule_check_greater(self):
        """Test > operator."""
        rule = AlertRule(
            metric_name="latency",
            operator=">",
            threshold=100.0,
            severity="critical",
            message="Test",
        )
        
        assert rule.check(150.0) is True
        assert rule.check(100.0) is False
        assert rule.check(50.0) is False
    
    def test_alert_rule_check_less(self):
        """Test < operator."""
        rule = AlertRule(
            metric_name="rate",
            operator="<",
            threshold=0.5,
            severity="warning",
            message="Test",
        )
        
        assert rule.check(0.3) is True
        assert rule.check(0.5) is False
        assert rule.check(0.7) is False
    
    def test_alert_rule_invalid_operator(self):
        """Test invalid operator."""
        rule = AlertRule(
            metric_name="test",
            operator="invalid",
            threshold=0.5,
            severity="warning",
            message="Test",
        )
        
        with pytest.raises(ValueError, match="Unknown operator"):
            rule.check(0.5)
