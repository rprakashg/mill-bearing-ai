"""Command-line training: python -m bearing.train --out models --version 1 [--upload] [--register]

Produces models/bearing/<version>/model.onnx (OVMS layout) and metrics.json.
Used for the laptop/edge quick start; the notebooks and pipeline call the same functions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MODEL_NAME
from .model import export_onnx, train, verify_onnx
from .simulate import make_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    ap.add_argument("--version", default="1")
    ap.add_argument("--n-per-class", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-f1", type=float, default=0.90, help="evaluation gate")
    ap.add_argument("--upload", action="store_true", help="upload to S3 (AWS_* env vars)")
    ap.add_argument("--register", action="store_true", help="register in the OpenShift AI Model Registry")
    a = ap.parse_args()

    X, y = make_dataset(a.n_per_class, a.seed)
    clf, metrics = train(X, y, a.seed)
    print(metrics["report"])
    if metrics["macro_f1"] < a.min_f1:
        raise SystemExit(f"Evaluation gate failed: macro F1 {metrics['macro_f1']} < {a.min_f1}")

    path = export_onnx(clf, Path(a.out) / MODEL_NAME / a.version / "model.onnx")
    print(f"ONNX vs scikit-learn max diff: {verify_onnx(path, clf, X[:256]):.2e}")
    (Path(a.out) / f"{MODEL_NAME}-metrics.json").write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print(f"wrote {path}")

    if a.upload:
        from .storage import upload_model
        uri = upload_model(path, a.version)
        print(f"uploaded to {uri}")
        if a.register:
            from .registry import register
            register(uri, a.version, metrics)


if __name__ == "__main__":
    main()
