# AUDIT Expert - Quality Monitoring System

## Overview

AUDIT (Automated Quality Inspection and Diagnostics) Expert monitors memory system quality by analyzing shadow traces and computing key metrics.

## Architecture

```
shadow_traces.db
    ↓
AuditExpert
    ↓
QualityMetrics → AlertRule → Alert
    ↓
Report (JSON)
```

## Core Metrics

1. **citation_rate**: Proportion of retrievals that result in citations (higher is better)
2. **correction_rate**: Proportion of citations that get corrected (lower is better)
3. **context_pollution**: Proportion of injected memories ignored (lower is better)
4. **retrieval_latency_p99**: 99th percentile retrieval latency in ms (lower is better)
5. **memory_growth_rate**: New memories created per day
6. **unknown_ratio**: Proportion of queries with no relevant memories

## Default Alert Rules

| Metric | Threshold | Severity | Condition |
|--------|-----------|----------|-----------|
| citation_rate | < 30% | warning | Too few memories are being used |
| correction_rate | > 15% | critical | Too many memories are incorrect |
| context_pollution | > 10% | warning | Too many irrelevant injected memories |
| retrieval_latency_p99 | > 600ms | critical | System is too slow |

## Usage

### Command Line

```bash
# Basic usage
python scripts/run_audit.py --db shadow_traces.db --output logs/audit

# 14-day window
python scripts/run_audit.py --db shadow_traces.db --window-days 14

# Compare with baseline
python scripts/run_audit.py --db shadow_traces.db --baseline logs/audit/baseline.json

# Alert-only mode (for monitoring)
python scripts/run_audit.py --db shadow_traces.db --alert-only

# JSON output (for automation)
python scripts/run_audit.py --db shadow_traces.db --json-output
```

### Python API

```python
from audit_expert import AuditExpert, AlertRule

# Initialize
expert = AuditExpert(db_path="shadow_traces.db")

# Compute metrics
metrics = expert.compute_metrics(window_days=7)
print(f"Citation rate: {metrics.citation_rate:.2%}")

# Check alerts
alerts = expert.check_alerts(metrics)
for alert in alerts:
    print(f"{alert.rule.severity}: {alert.rule.message}")

# Generate report
report = expert.generate_report(
    window_days=7,
    output_path="logs/audit/report.json"
)

# Custom alert rules
custom_rules = [
    AlertRule(
        metric_name="citation_rate",
        operator="<",
        threshold=0.5,
        severity="critical",
        message="Citation rate critically low"
    )
]
expert = AuditExpert(db_path="shadow_traces.db", alert_rules=custom_rules)
```

### Baseline Comparison

```python
from audit_expert import compare_with_baseline

# Load metrics
current_metrics = expert.compute_metrics(window_days=7)
baseline_metrics = expert.compute_metrics(window_days=7, end_time=baseline_timestamp)

# Compare
comparison = compare_with_baseline(current_metrics, baseline_metrics)

for metric_name, result in comparison.items():
    if result["significant"]:
        print(f"{metric_name}: {result['delta']:+.4f} (p={result['p_improvement']:.3f})")
```

## Scheduling

### Linux/Mac (cron)

```bash
# Every 6 hours
0 */6 * * * cd /path/to/MNEMOSYNE-MNS && python scripts/run_audit.py --db shadow_traces.db --output logs/audit --alert-only

# Daily at 2 AM with baseline comparison
0 2 * * * cd /path/to/MNEMOSYNE-MNS && python scripts/run_audit.py --db shadow_traces.db --baseline logs/audit/baseline.json --output logs/audit
```

### Windows (Task Scheduler)

```powershell
schtasks /create /tn "MNEMOSYNE Audit" /tr "python D:\project\MNEMOSYNE-MNS\scripts\run_audit.py --db shadow_traces.db --output logs/audit" /sc hourly
```

### Docker

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Healthcheck using audit
HEALTHCHECK --interval=1h --timeout=30s \
  CMD python scripts/run_audit.py --db shadow_traces.db --alert-only || exit 1
```

## Exit Codes

- `0`: Healthy (no alerts)
- `1`: Warning (non-critical alerts)
- `2`: Critical (critical alerts)

## Report Format

```json
{
  "generated_at": 1793849395.789,
  "window_days": 7,
  "metrics": {
    "window_start": 1793244595.787,
    "window_end": 1793849395.787,
    "metrics": {
      "citation_rate": 0.72,
      "correction_rate": 0.08,
      "context_pollution": 0.05,
      "retrieval_latency_p99": 245.3,
      "memory_growth_rate": 12.5,
      "unknown_ratio": 0.03
    },
    "counts": {
      "total_retrievals": 1523,
      "total_citations": 1097,
      "total_corrections": 88,
      "total_ignored_injected": 15,
      "total_candidates": 7615
    }
  },
  "alerts": [
    {
      "metric": "correction_rate",
      "severity": "warning",
      "message": "Correction rate elevated - review memory quality",
      "threshold": 0.05,
      "operator": ">",
      "actual_value": 0.08,
      "timestamp": 1793849395.789
    }
  ],
  "status": "warning",
  "baseline_comparison": {
    "citation_rate": {
      "current": 0.72,
      "baseline": 0.68,
      "delta": 0.04,
      "ci95_lower": 0.02,
      "ci95_upper": 0.06,
      "p_improvement": 0.998,
      "significant": true
    }
  }
}
```

## Integration

### CI/CD Pipeline

```yaml
# .github/workflows/audit.yml
name: Quality Audit
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Audit
        run: |
          python scripts/run_audit.py --db shadow_traces.db --alert-only
      - name: Upload Report
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: audit-report
          path: logs/audit/
```

### Alert Notifications

```python
# Example: Send Slack notification on critical alerts
import requests
import json

report = expert.generate_report(window_days=7)

if report["status"] == "critical":
    critical_alerts = [a for a in report["alerts"] if a["severity"] == "critical"]
    
    message = {
        "text": f"🚨 MNEMOSYNE Critical Quality Alert",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{len(critical_alerts)} critical alert(s) detected*"
                }
            }
        ]
    }
    
    for alert in critical_alerts:
        message["blocks"].append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"• {alert['message']}\n  Actual: {alert['actual_value']:.4f}, Threshold: {alert['threshold']}"
            }
        })
    
    requests.post(SLACK_WEBHOOK_URL, json=message)
```

## Files

- `audit_expert.py`: Core monitoring module
- `scripts/run_audit.py`: CLI for scheduled monitoring
- `tests/test_audit_expert.py`: Comprehensive test suite (17 tests, 100% pass)
- `reports/audit_validation_report.json`: Validation and deployment guide

## Testing

```bash
# Run all tests
python -m pytest tests/test_audit_expert.py -v

# Run specific test
python -m pytest tests/test_audit_expert.py::TestAuditExpert::test_compute_metrics_healthy -v

# With coverage
python -m pytest tests/test_audit_expert.py --cov=audit_expert --cov-report=html
```

## Next Steps

1. **Establish Baseline**: Run initial audit on production data to set baseline metrics
2. **Schedule Monitoring**: Set up cron job or scheduled task to run every 6 hours
3. **Configure Alerts**: Adjust thresholds based on system characteristics
4. **Integrate Notifications**: Connect to Slack, email, or PagerDuty for real-time alerts
5. **Trend Analysis**: Collect reports over time to identify trends and patterns
6. **CI/CD Integration**: Block deployments when critical quality alerts are triggered
