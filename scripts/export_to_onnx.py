"""
Export RANK-HYBRID-01 to ONNX format for C++ deployment.
"""

import torch
import joblib
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


def export_e5_to_onnx(output_path: Path):
    """Export E5 encoder to ONNX."""
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    
    # Dummy input
    dummy_text = "测试文本"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    
    # Export
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(output_path / "e5_encoder.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"}
        },
        opset_version=14
    )
    
    # Save tokenizer vocab
    tokenizer.save_pretrained(str(output_path / "tokenizer"))
    print(f"✓ E5 encoder exported to {output_path / 'e5_encoder.onnx'}")


def export_ranker_to_onnx(model_path: Path, output_path: Path):
    """Export logistic ranker to ONNX."""
    # Load trained ranker
    artifacts = joblib.load(model_path / "hybrid_ranker.joblib")
    scaler = artifacts["scaler"]
    ranker = artifacts["ranker"]
    
    # Save scaler parameters as JSON
    import json
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    (output_path / "scaler.json").write_text(json.dumps(scaler_params, indent=2))
    
    # Save ranker weights
    ranker_params = {
        "coef": ranker.coef_[0].tolist(),
        "intercept": float(ranker.intercept_[0]),
    }
    (output_path / "ranker.json").write_text(json.dumps(ranker_params, indent=2))
    
    print(f"✓ Ranker parameters exported to {output_path}")


def main():
    output_dir = Path("runs/onnx_export")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Exporting MNEMOSYNE-MNS to ONNX format...\n")
    
    export_e5_to_onnx(output_dir)
    export_ranker_to_onnx(Path("runs/rank_hybrid_v1"), output_dir)
    
    print("\n✓ Export complete!")
    print(f"Files: {list(output_dir.glob('*'))}")


if __name__ == "__main__":
    main()
