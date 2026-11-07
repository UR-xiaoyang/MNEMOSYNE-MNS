#!/usr/bin/env python3
"""
Run AUDIT quality monitoring and generate reports.

This script can be run as a cron job or scheduled task to monitor
memory system quality continuously.

Usage:
    python scripts/run_audit.py --db shadow_traces.db --output logs/audit/
    python scripts/run_audit.py --db shadow_traces.db --window-days 14
    python scripts/run_audit.py --db shadow_traces.db --baseline logs/audit/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from audit_expert import AuditExpert, QualityMetrics, compare_with_baseline


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run AUDIT quality monitoring")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("shadow_traces.db"),
        help="Path to shadow_traces.db (default: shadow_traces.db)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/audit"),
        help="Output directory for reports (default: logs/audit)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Sliding window size in days (default: 7)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline report to compare against",
    )
    parser.add_argument(
        "--alert-only",
        action="store_true",
        help="Only print alerts, not full report",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output as JSON instead of human-readable format",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.db.exists():
        print(f"❌ Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Initialize AUDIT expert
    try:
        expert = AuditExpert(db_path=args.db)
    except Exception as e:
        print(f"❌ Failed to initialize AUDIT expert: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output / f"audit_report_{timestamp}.json"
    
    try:
        report = expert.generate_report(
            window_days=args.window_days,
            output_path=report_path,
        )
    except Exception as e:
        print(f"❌ Failed to generate report: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Compare with baseline if provided
    comparison_results = None
    if args.baseline:
        if not args.baseline.exists():
            print(f"⚠️  Baseline not found: {args.baseline}", file=sys.stderr)
        else:
            try:
                baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
                baseline_metrics_dict = baseline_report["metrics"]
                
                # Reconstruct QualityMetrics from baseline
                baseline_metrics = QualityMetrics(
                    window_start=baseline_metrics_dict["window_start"],
                    window_end=baseline_metrics_dict["window_end"],
                    citation_rate=baseline_metrics_dict["metrics"]["citation_rate"],
                    correction_rate=baseline_metrics_dict["metrics"]["correction_rate"],
                    context_pollution=baseline_metrics_dict["metrics"]["context_pollution"],
                    retrieval_latency_p99=baseline_metrics_dict["metrics"]["retrieval_latency_p99"],
                    memory_growth_rate=baseline_metrics_dict["metrics"]["memory_growth_rate"],
                    unknown_ratio=baseline_metrics_dict["metrics"]["unknown_ratio"],
                    total_retrievals=baseline_metrics_dict["counts"]["total_retrievals"],
                    total_citations=baseline_metrics_dict["counts"]["total_citations"],
                    total_corrections=baseline_metrics_dict["counts"]["total_corrections"],
                    total_ignored_injected=baseline_metrics_dict["counts"]["total_ignored_injected"],
                    total_candidates=baseline_metrics_dict["counts"]["total_candidates"],
                )
                
                current_metrics = QualityMetrics(
                    window_start=report["metrics"]["window_start"],
                    window_end=report["metrics"]["window_end"],
                    citation_rate=report["metrics"]["metrics"]["citation_rate"],
                    correction_rate=report["metrics"]["metrics"]["correction_rate"],
                    context_pollution=report["metrics"]["metrics"]["context_pollution"],
                    retrieval_latency_p99=report["metrics"]["metrics"]["retrieval_latency_p99"],
                    memory_growth_rate=report["metrics"]["metrics"]["memory_growth_rate"],
                    unknown_ratio=report["metrics"]["metrics"]["unknown_ratio"],
                    total_retrievals=report["metrics"]["counts"]["total_retrievals"],
                    total_citations=report["metrics"]["counts"]["total_citations"],
                    total_corrections=report["metrics"]["counts"]["total_corrections"],
                    total_ignored_injected=report["metrics"]["counts"]["total_ignored_injected"],
                    total_candidates=report["metrics"]["counts"]["total_candidates"],
                )
                
                comparison_results = compare_with_baseline(current_metrics, baseline_metrics)
                
                # Add comparison to report
                report["baseline_comparison"] = comparison_results
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                
            except Exception as e:
                print(f"⚠️  Failed to compare with baseline: {e}", file=sys.stderr)
    
    # Output results
    if args.json_output:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(0)
    
    # Human-readable output
    if args.alert_only:
        if report["alerts"]:
            print(f"🚨 {len(report['alerts'])} alert(s) triggered:")
            for alert in report["alerts"]:
                emoji = "🔴" if alert["severity"] == "critical" else "⚠️"
                print(f"\n{emoji} {alert['severity'].upper()}: {alert['message']}")
                print(f"   Metric: {alert['metric']}")
                print(f"   Threshold: {alert['operator']} {alert['threshold']}")
                print(f"   Actual: {alert['actual_value']:.4f}")
        else:
            print("✅ No alerts triggered - system is healthy")
    else:
        # Full report
        print(f"\n{'='*70}")
        print(f"AUDIT Quality Monitoring Report")
        print(f"{'='*70}")
        print(f"Generated: {datetime.fromtimestamp(report['generated_at']).isoformat()}")
        print(f"Window: {args.window_days} days")
        print(f"Status: {report['status'].upper()}")
        print(f"Report saved: {report_path}")
        
        print(f"\n{'='*70}")
        print("Quality Metrics")
        print(f"{'='*70}")
        metrics = report["metrics"]["metrics"]
        counts = report["metrics"]["counts"]
        
        print(f"Citation Rate:        {metrics['citation_rate']:.2%} ({counts['total_citations']} / {counts['total_retrievals']} retrievals)")
        print(f"Correction Rate:      {metrics['correction_rate']:.2%} ({counts['total_corrections']} / {counts['total_citations']} citations)")
        print(f"Context Pollution:    {metrics['context_pollution']:.2%} ({counts['total_ignored_injected']} / {counts['total_candidates']} candidates)")
        print(f"Retrieval Latency P99: {metrics['retrieval_latency_p99']:.1f}ms")
        print(f"Memory Growth Rate:   {metrics['memory_growth_rate']:.1f} memories/day")
        print(f"Unknown Ratio:        {metrics['unknown_ratio']:.2%}")
        
        if report["alerts"]:
            print(f"\n{'='*70}")
            print(f"Alerts ({len(report['alerts'])})")
            print(f"{'='*70}")
            for alert in report["alerts"]:
                emoji = "🔴" if alert["severity"] == "critical" else "⚠️"
                print(f"\n{emoji} {alert['severity'].upper()}: {alert['message']}")
                print(f"   Metric: {alert['metric']}")
                print(f"   Threshold: {alert['operator']} {alert['threshold']}")
                print(f"   Actual: {alert['actual_value']:.4f}")
        
        if comparison_results:
            print(f"\n{'='*70}")
            print("Baseline Comparison")
            print(f"{'='*70}")
            for metric_name, result in comparison_results.items():
                delta = result["delta"]
                significant = result["significant"]
                p_improvement = result["p_improvement"]
                
                # Determine if change is good or bad
                better_when_higher = metric_name in ["citation_rate", "memory_growth_rate"]
                better_when_lower = metric_name in ["correction_rate", "context_pollution", "retrieval_latency_p99", "unknown_ratio"]
                
                if better_when_higher:
                    direction = "↑" if delta > 0 else "↓"
                    is_improvement = delta > 0
                elif better_when_lower:
                    direction = "↓" if delta < 0 else "↑"
                    is_improvement = delta < 0
                else:
                    direction = "→"
                    is_improvement = False
                
                emoji = "✅" if is_improvement and significant else "⚠️" if significant else "→"
                
                print(f"{emoji} {metric_name:25} {direction} {delta:+.4f} (p={p_improvement:.3f}, {'sig' if significant else 'ns'})")
                print(f"   Current: {result['current']:.4f}, Baseline: {result['baseline']:.4f}")
                print(f"   95% CI: [{result['ci95_lower']:.4f}, {result['ci95_upper']:.4f}]")
        
        print(f"\n{'='*70}\n")
    
    # Exit with error code if critical alerts
    if any(a["severity"] == "critical" for a in report["alerts"]):
        sys.exit(2)
    elif report["alerts"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
