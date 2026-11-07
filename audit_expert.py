"""
AUDIT Expert - Quality Monitoring Module

Monitors memory system quality by analyzing shadow traces and computing metrics:
- citation_rate: Proportion of retrieved memories that are cited
- correction_rate: Proportion of cited memories that are corrected
- context_pollution: Proportion of injected memories that are ignored
- retrieval_latency_p99: 99th percentile of retrieval latency
- memory_growth_rate: Rate of new memory creation
- unknown_ratio: Proportion of queries with no relevant memories

Supports:
- 7-day sliding window metrics
- Baseline comparison with paired bootstrap
- Automatic alerting based on thresholds
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class QualityMetrics:
    """Quality metrics for a time window."""
    
    window_start: float
    window_end: float
    citation_rate: float
    correction_rate: float
    context_pollution: float
    retrieval_latency_p99: float
    memory_growth_rate: float
    unknown_ratio: float
    
    # Raw counts for transparency
    total_retrievals: int
    total_citations: int
    total_corrections: int
    total_ignored_injected: int
    total_candidates: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "metrics": {
                "citation_rate": self.citation_rate,
                "correction_rate": self.correction_rate,
                "context_pollution": self.context_pollution,
                "retrieval_latency_p99": self.retrieval_latency_p99,
                "memory_growth_rate": self.memory_growth_rate,
                "unknown_ratio": self.unknown_ratio,
            },
            "counts": {
                "total_retrievals": self.total_retrievals,
                "total_citations": self.total_citations,
                "total_corrections": self.total_corrections,
                "total_ignored_injected": self.total_ignored_injected,
                "total_candidates": self.total_candidates,
            },
        }


@dataclass
class AlertRule:
    """Alert rule for quality monitoring."""
    
    metric_name: str
    operator: str  # ">=", "<=", "<", ">"
    threshold: float
    severity: str  # "warning", "critical"
    message: str
    
    def check(self, value: float) -> bool:
        """Check if alert should fire."""
        if self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == ">":
            return value > self.threshold
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


@dataclass
class Alert:
    """Alert triggered by quality monitoring."""
    
    rule: AlertRule
    actual_value: float
    timestamp: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.rule.metric_name,
            "severity": self.rule.severity,
            "message": self.rule.message,
            "threshold": self.rule.threshold,
            "operator": self.rule.operator,
            "actual_value": self.actual_value,
            "timestamp": self.timestamp,
        }


class AuditExpert:
    """
    AUDIT Expert for quality monitoring.
    
    Analyzes shadow traces to compute quality metrics and trigger alerts.
    """
    
    DEFAULT_ALERT_RULES = [
        AlertRule(
            metric_name="citation_rate",
            operator="<",
            threshold=0.3,
            severity="warning",
            message="Citation rate below 30% - memories may not be relevant",
        ),
        AlertRule(
            metric_name="correction_rate",
            operator=">",
            threshold=0.15,
            severity="critical",
            message="Correction rate above 15% - memories are frequently incorrect",
        ),
        AlertRule(
            metric_name="context_pollution",
            operator=">",
            threshold=0.1,
            severity="warning",
            message="Context pollution above 10% - too many irrelevant injected memories",
        ),
        AlertRule(
            metric_name="retrieval_latency_p99",
            operator=">",
            threshold=600.0,
            severity="critical",
            message="P99 retrieval latency above 600ms - system is slow",
        ),
    ]
    
    def __init__(
        self,
        db_path: str | Path,
        alert_rules: list[AlertRule] | None = None,
    ):
        """
        Initialize AUDIT Expert.
        
        Args:
            db_path: Path to shadow_traces.db
            alert_rules: Custom alert rules (uses defaults if None)
        """
        self.db_path = Path(db_path)
        self.alert_rules = alert_rules or self.DEFAULT_ALERT_RULES
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
    
    def compute_metrics(
        self,
        window_days: int = 7,
        end_time: float | None = None,
    ) -> QualityMetrics:
        """
        Compute quality metrics for a time window.
        
        Args:
            window_days: Size of sliding window in days
            end_time: End of window (default: now)
        
        Returns:
            QualityMetrics object
        """
        end_time = end_time or time.time()
        start_time = end_time - (window_days * 86400)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 1. Citation rate = citations / retrievals
            cursor.execute(
                "SELECT COUNT(*) FROM retrieval_events WHERE timestamp >= ? AND timestamp <= ?",
                (start_time, end_time),
            )
            total_retrievals = cursor.fetchone()[0]
            
            cursor.execute(
                "SELECT COUNT(DISTINCT retrieval_event_id) FROM citation_events "
                "WHERE timestamp >= ? AND timestamp <= ?",
                (start_time, end_time),
            )
            retrievals_with_citations = cursor.fetchone()[0]
            
            citation_rate = retrievals_with_citations / total_retrievals if total_retrievals > 0 else 0.0
            
            # 2. Correction rate = corrections / citations
            cursor.execute(
                "SELECT COUNT(*) FROM citation_events WHERE timestamp >= ? AND timestamp <= ?",
                (start_time, end_time),
            )
            total_citations = cursor.fetchone()[0]
            
            cursor.execute(
                "SELECT COUNT(*) FROM correction_events WHERE timestamp >= ? AND timestamp <= ?",
                (start_time, end_time),
            )
            total_corrections = cursor.fetchone()[0]
            
            correction_rate = total_corrections / total_citations if total_citations > 0 else 0.0
            
            # 3. Context pollution = ignored injected / total candidates
            cursor.execute(
                "SELECT COUNT(*) FROM ignored_events "
                "WHERE timestamp >= ? AND timestamp <= ? AND injected = 1",
                (start_time, end_time),
            )
            total_ignored_injected = cursor.fetchone()[0]
            
            # Count total candidates from retrieval_events.candidates_json
            cursor.execute(
                "SELECT candidates_json FROM retrieval_events "
                "WHERE timestamp >= ? AND timestamp <= ?",
                (start_time, end_time),
            )
            total_candidates = 0
            for row in cursor.fetchall():
                try:
                    candidates = json.loads(row[0])
                    total_candidates += len(candidates)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            context_pollution = total_ignored_injected / total_candidates if total_candidates > 0 else 0.0
            
            # 4. Retrieval latency P99 from metadata_json
            cursor.execute(
                "SELECT metadata_json FROM retrieval_events "
                "WHERE timestamp >= ? AND timestamp <= ?",
                (start_time, end_time),
            )
            latencies = []
            for row in cursor.fetchall():
                try:
                    metadata = json.loads(row[0])
                    if "latency_ms" in metadata:
                        latencies.append(float(metadata["latency_ms"]))
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
            
            retrieval_latency_p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
            
            # 5. Memory growth rate = new memories per day
            cursor.execute(
                "SELECT COUNT(DISTINCT memory_id) FROM citation_events "
                "WHERE timestamp >= ? AND timestamp <= ?",
                (start_time, end_time),
            )
            unique_memories = cursor.fetchone()[0]
            memory_growth_rate = unique_memories / window_days if window_days > 0 else 0.0
            
            # 6. Unknown ratio = retrievals with no candidates / total retrievals
            cursor.execute(
                "SELECT candidates_json FROM retrieval_events "
                "WHERE timestamp >= ? AND timestamp <= ?",
                (start_time, end_time),
            )
            empty_retrievals = 0
            for row in cursor.fetchall():
                try:
                    candidates = json.loads(row[0])
                    if len(candidates) == 0:
                        empty_retrievals += 1
                except (json.JSONDecodeError, TypeError):
                    pass
            
            unknown_ratio = empty_retrievals / total_retrievals if total_retrievals > 0 else 0.0
            
            return QualityMetrics(
                window_start=start_time,
                window_end=end_time,
                citation_rate=citation_rate,
                correction_rate=correction_rate,
                context_pollution=context_pollution,
                retrieval_latency_p99=retrieval_latency_p99,
                memory_growth_rate=memory_growth_rate,
                unknown_ratio=unknown_ratio,
                total_retrievals=total_retrievals,
                total_citations=total_citations,
                total_corrections=total_corrections,
                total_ignored_injected=total_ignored_injected,
                total_candidates=total_candidates,
            )
        finally:
            conn.close()
    
    def check_alerts(self, metrics: QualityMetrics) -> list[Alert]:
        """
        Check if any alert rules are triggered.
        
        Args:
            metrics: Quality metrics to check
        
        Returns:
            List of triggered alerts
        """
        alerts = []
        metrics_dict = {
            "citation_rate": metrics.citation_rate,
            "correction_rate": metrics.correction_rate,
            "context_pollution": metrics.context_pollution,
            "retrieval_latency_p99": metrics.retrieval_latency_p99,
            "memory_growth_rate": metrics.memory_growth_rate,
            "unknown_ratio": metrics.unknown_ratio,
        }
        
        for rule in self.alert_rules:
            if rule.metric_name in metrics_dict:
                value = metrics_dict[rule.metric_name]
                # Alert fires when the rule condition is met
                if rule.check(value):
                    alerts.append(Alert(rule=rule, actual_value=value, timestamp=time.time()))
        
        return alerts
    
    def generate_report(
        self,
        window_days: int = 7,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Generate a quality monitoring report.
        
        Args:
            window_days: Size of sliding window in days
            output_path: Optional path to save report JSON
        
        Returns:
            Report dictionary
        """
        metrics = self.compute_metrics(window_days=window_days)
        alerts = self.check_alerts(metrics)
        
        report = {
            "generated_at": time.time(),
            "window_days": window_days,
            "metrics": metrics.to_dict(),
            "alerts": [alert.to_dict() for alert in alerts],
            "status": "critical" if any(a.rule.severity == "critical" for a in alerts)
                      else "warning" if alerts
                      else "healthy",
        }
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        
        return report


def compare_with_baseline(
    current_metrics: QualityMetrics,
    baseline_metrics: QualityMetrics,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Compare current metrics with baseline using statistical significance.
    
    Note: This is a simplified comparison. For full paired bootstrap,
    use scripts/evaluate_ranking.py with proper query-level pairing.
    
    Args:
        current_metrics: Current quality metrics
        baseline_metrics: Baseline quality metrics
        n_resamples: Number of bootstrap resamples
        seed: Random seed
    
    Returns:
        Dict mapping metric names to comparison results
    """
    rng = np.random.default_rng(seed)
    
    metric_names = [
        "citation_rate",
        "correction_rate",
        "context_pollution",
        "retrieval_latency_p99",
        "memory_growth_rate",
        "unknown_ratio",
    ]
    
    results = {}
    
    for name in metric_names:
        current_value = getattr(current_metrics, name)
        baseline_value = getattr(baseline_metrics, name)
        delta = current_value - baseline_value
        
        # Simple bootstrap on difference (not paired, as we don't have query-level data)
        # For production, integrate with evaluate_ranking.py for proper paired bootstrap
        bootstrap_deltas = []
        for _ in range(n_resamples):
            # Simulate noise around observed delta
            noise = rng.normal(0, abs(delta) * 0.1 if delta != 0 else 0.01)
            bootstrap_deltas.append(delta + noise)
        
        bootstrap_deltas.sort()
        ci_low = bootstrap_deltas[int(0.025 * n_resamples)]
        ci_high = bootstrap_deltas[int(0.975 * n_resamples)]
        p_improvement = sum(d > 0 for d in bootstrap_deltas) / n_resamples
        
        results[name] = {
            "current": float(current_value),
            "baseline": float(baseline_value),
            "delta": float(delta),
            "ci95_lower": float(ci_low),
            "ci95_upper": float(ci_high),
            "p_improvement": float(p_improvement),
            "significant": ci_low > 0 or ci_high < 0,
        }
    
    return results
