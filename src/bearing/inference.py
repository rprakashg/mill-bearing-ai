"""Scoring clients with one interface: KServe v2 REST (OVMS on OpenShift AI or the edge)
or a local ONNX Runtime session (laptop fallback, no server needed)."""
from __future__ import annotations

import json
import ssl
import urllib.request

import numpy as np

from .config import CLASSES, MODEL_NAME, N_FEATURES, ONNX_INPUT, ONNX_OUTPUT


class KServeV2Client:
    """POST {url}/v2/models/{model}/infer with the Open Inference Protocol (KServe v2)."""

    def __init__(self, base_url: str, model: str = MODEL_NAME, token: str | None = None,
                 verify_tls: bool = True, timeout: float = 5.0):
        self.url = f"{base_url.rstrip('/')}/v2/models/{model}/infer"
        self.token, self.timeout = token, timeout
        self.ctx = None if verify_tls else ssl._create_unverified_context()

    def predict_proba(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        body = {"inputs": [{"name": ONNX_INPUT, "shape": list(X.shape), "datatype": "FP32",
                            "data": X.flatten().tolist()}]}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=self.timeout, context=self.ctx) as r:
            out = json.loads(r.read())
        o = next((o for o in out["outputs"] if o["name"] == ONNX_OUTPUT), out["outputs"][0])
        return np.asarray(o["data"], dtype=np.float32).reshape(-1, len(CLASSES))


class LocalOnnxClient:
    def __init__(self, path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def predict_proba(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32)).reshape(-1, N_FEATURES)
        return self.sess.run([ONNX_OUTPUT], {ONNX_INPUT: X})[0]


def label(probs: np.ndarray) -> list[tuple[str, float]]:
    return [(CLASSES[int(i)], float(p[int(i)])) for p, i in zip(probs, probs.argmax(axis=1))]
