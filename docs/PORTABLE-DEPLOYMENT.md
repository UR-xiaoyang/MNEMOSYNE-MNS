# Portable Deployment Notes

## What is now fixed

The deployable Hybrid ranker now preserves the training-time TF-IDF vectorizer. The runtime also:

- resolves model paths from the project root rather than the caller's current directory;
- computes the 12 structured features when the caller omits them;
- preserves caller-supplied memory IDs through the API;
- supports offline E5 loading through `MNEMOSYNE_E5_MODEL`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`;
- validates the feature shape before scoring;
- keeps model weights and the E5 directory outside Git.

## Prepare a self-contained Docker build context

The private/local training artifacts are intentionally ignored by Git. Prepare them locally:

```bash
python3 scripts/prepare_deployment.py \
  --source runs/rank_hybrid_v1 \
  --vectorizer <path_to_features>/rerank_tfidf_vectorizer.joblib \
  --e5-source <path_to_local_e5_snapshot> \
  --destination deploy
```

This creates `deploy/model_bundle/`, `deploy/e5_model/`, and `deploy/manifest.json`. The manifest records every file's byte size and SHA-256. Verify it before copying the deployment directory to another host.

Then build:

```bash
docker build \
  --build-arg MODEL_BUNDLE=deploy/model_bundle \
  --build-arg E5_MODEL_DIR=deploy/e5_model \
  -t mnemosyne-mns:portable .
```

The build context contains model artifacts locally, but `deploy/`, `runs/`, and `models/` remain excluded from Git.

## Run without network access

```bash
docker run --rm -p 8000:8000 mnemosyne-mns:portable
curl http://localhost:8000/health
```

The image uses the local E5 directory and sets both Hugging Face offline flags. No model download is attempted at container startup.

## Verify before shipping

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py docker/server.py
```

The repository currently has a 5-test runtime/deployment contract suite. Docker image building requires a Docker daemon; C++ ONNX compilation requires ONNX Runtime headers/libs and is a separate gate.

## C++ status

`cpp/mnemosyne_inference.cpp` remains a reference ONNX Runtime integration. It is not claimed as a tested standalone binary because this checkout does not include ONNX Runtime headers, the SentencePiece tokenizer runtime, or a Docker daemon. Do not ship it as a self-contained executable until those dependencies are bundled and the binary is compiled and exercised on each target platform.
