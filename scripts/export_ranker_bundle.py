"""Export the small, non-neural ranker payload for deployment packaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def export_ranker(source: Path, destination: Path, vectorizer_path: Path | None = None) -> None:
    artifact = joblib.load(source / "hybrid_ranker.joblib")
    if artifact.get("vectorizer") is None:
        candidate = vectorizer_path or (source / "rerank_tfidf_vectorizer.joblib")
        if candidate.exists():
            artifact["vectorizer"] = joblib.load(candidate)
    if artifact.get("vectorizer") is None:
        raise FileNotFoundError(
            "A fitted TF-IDF vectorizer is required for hybrid deployment parity. "
            "Pass --vectorizer pointing to rerank_tfidf_vectorizer.joblib."
        )
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination / "hybrid_ranker.joblib")
    (destination / "scaler.json").write_text(
        json.dumps(
            {"mean": artifact["scaler"].mean_.tolist(), "scale": artifact["scaler"].scale_.tolist()},
            indent=2,
        ),
        encoding="utf-8",
    )
    (destination / "ranker.json").write_text(
        json.dumps({
            "coef": artifact["ranker"].coef_[0].tolist(),
            "intercept": float(artifact["ranker"].intercept_[0]),
            "feature_names": artifact["feature_names"],
            "model_id": "MNEMOSYNE-MNS-RANK-HYBRID-01",
            "e5_model": artifact["e5_model"],
        }, indent=2),
        encoding="utf-8",
    )
    (destination / "metadata.json").write_text(
        (source / "metadata.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("runs/rank_hybrid_v1"))
    parser.add_argument("--destination", type=Path, default=Path("deploy/model_bundle"))
    parser.add_argument("--vectorizer", type=Path, default=None)
    args = parser.parse_args()
    export_ranker(args.source, args.destination, args.vectorizer)
    print(f"Exported deployment bundle to {args.destination.resolve()}")


if __name__ == "__main__":
    main()
