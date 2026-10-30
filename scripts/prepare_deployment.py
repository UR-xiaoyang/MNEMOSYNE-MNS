"""Prepare a self-contained, hash-manifested deployment directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from export_ranker_bundle import export_ranker


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_deployment(
    source: Path,
    vectorizer_path: Path,
    e5_source: Path,
    destination: Path,
) -> dict:
    """Prepare ``model_bundle``, ``e5_model`` and a reproducibility manifest."""
    destination = destination.resolve()
    bundle = destination / "model_bundle"
    e5_destination = destination / "e5_model"
    if destination.exists():
        shutil.rmtree(destination)
    bundle.mkdir(parents=True)
    e5_destination.mkdir(parents=True)

    export_ranker(source, bundle, vectorizer_path)
    for item in e5_source.iterdir():
        if item.is_file():
            shutil.copy2(item, e5_destination / item.name)

    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    files = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(destination).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    manifest = {
        "model_id": metadata["model_id"],
        "encoder": metadata.get("components", {}).get("semantic_encoder", "unknown"),
        "files": files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("runs/rank_hybrid_v1"))
    parser.add_argument("--vectorizer", type=Path, required=True)
    parser.add_argument("--e5-source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("deploy"))
    args = parser.parse_args()
    manifest = prepare_deployment(args.source, args.vectorizer, args.e5_source, args.destination)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
