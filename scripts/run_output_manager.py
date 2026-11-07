"""
Run output directory management and manifest generation.

Provides atomic output directory preparation and reproducible manifest construction
for training runs, with no dependency on heavy ML frameworks.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Any


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare_output_directory(output_dir: Path, overwrite: bool) -> dict[str, Any] | None:
    """
    Prepare output directory with atomic isolation guarantees.

    Args:
        output_dir: Target output directory path
        overwrite: If True, clear existing directory; if False, reject if exists

    Returns:
        Overwrite audit dict if overwrite was performed, None otherwise

    Raises:
        FileExistsError: If directory exists and overwrite=False
    """
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{output_dir}' already exists. "
                f"Use --overwrite to explicitly allow overwriting, or choose a different output directory."
            )

        # Overwrite mode: clear all files but preserve directory
        removed_files = []
        for item in output_dir.iterdir():
            if item.is_file():
                removed_files.append(item.name)
                item.unlink()
            elif item.is_dir():
                removed_files.append(f"{item.name}/")
                shutil.rmtree(item)

        audit = {
            "overwritten_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "removed_files": removed_files,
            "policy_version": "clear_on_overwrite_v1",
        }
        return audit
    else:
        # New directory: create it
        output_dir.mkdir(parents=True, exist_ok=False)
        return None


def build_training_manifest(
    run_id: str,
    seed: int,
    input_hashes: dict[str, str],
    counts: dict[str, int],
    elapsed_seconds: float,
    overwrite_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Build training run manifest with reproducibility metadata.

    Args:
        run_id: Unique run identifier
        seed: Random seed used
        input_hashes: Dict mapping input names to SHA-256 hashes
        counts: Dict with train_rows, val_rows, train_queries, val_queries, train_pairs
        elapsed_seconds: Training duration
        overwrite_audit: Overwrite audit if directory was cleared, None otherwise

    Returns:
        Complete manifest dict ready for JSON serialization
    """
    manifest = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "model_id": "MNEMOSYNE-MNS-RANK-HYBRID-01",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": seed,
        "input_hashes": input_hashes,
        "train_queries": counts["train_queries"],
        "validation_queries": counts["val_queries"],
        "train_candidate_count": counts["train_rows"],
        "validation_candidate_count": counts["val_rows"],
        "train_pair_count": counts["train_pairs"],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "overwrite_audit": overwrite_audit,
    }

    return manifest
