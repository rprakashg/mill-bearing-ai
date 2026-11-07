import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from bearing.config import CLASSES, N_FEATURES
from bearing.edge.scorer import Scorer, rule_based
from bearing.features import extract_features
from bearing.inference import KServeV2Client, LocalOnnxClient
from bearing.model import export_onnx, train, verify_onnx
from bearing.simulate import make_dataset, simulate_window


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    X, y = make_dataset(300, seed=3)
    clf, metrics = train(X, y, seed=3)
    path = export_onnx(clf, tmp_path_factory.mktemp("m") / "bearing" / "1" / "model.onnx")
    return clf, metrics, str(path), X


def window_msg(mode, sev, rng):
    w = simulate_window(mode, sev, rng)
    return {"asset": "planer-3", "samples": w["samples"].tolist(), "temp_c": w["temp_c"],
            "current_a": w["current_a"], "sample_rate_hz": w["sample_rate_hz"], "truth": mode}


def test_features_have_expected_shape():
    w = simulate_window("bearing_wear", 0.8)
    assert extract_features(w["samples"], w["temp_c"], w["current_a"]).shape == (N_FEATURES,)


def test_model_quality_and_onnx_parity(trained):
    clf, metrics, path, X = trained
    assert metrics["macro_f1"] >= 0.9
    assert verify_onnx(path, clf, X[:200]) < 1e-4


def test_scorer_debounces_then_clears(trained):
    _, _, path, _ = trained
    alerts = []
    s = Scorer(LocalOnnxClient(path), threshold=0.7, consecutive=3, publish=lambda t, p: alerts.append(json.loads(p)))
    rng = np.random.default_rng(5)
    for _ in range(2):
        s.handle(window_msg("bearing_wear", 0.9, rng))
    assert alerts == []                       # not yet: needs 3 in a row
    for _ in range(3):
        s.handle(window_msg("bearing_wear", 0.9, rng))
    assert len(alerts) == 1 and alerts[0]["label"] == "bearing_wear" and alerts[0]["severity"] == "high"
    s.handle(window_msg("healthy", 0.0, rng))
    assert alerts[-1]["severity"] == "cleared"


def test_scorer_falls_back_to_rules_when_model_down():
    class Down:
        def predict_proba(self, X):
            raise ConnectionError("model server unreachable")

    s = Scorer(Down(), consecutive=1, publish=lambda t, p: None)
    r = s.handle(window_msg("bearing_wear", 0.9, np.random.default_rng(1)))
    assert r["label"] == "bearing_wear" and r["source"].startswith("rules")


def test_rules_never_flag_healthy():
    X, y = make_dataset(200, seed=9)
    assert all(rule_based(x)[0] == "healthy" for x in X[y == 0])


def test_kserve_v2_client_round_trip(trained):
    """A fake OVMS speaking the Open Inference Protocol, backed by the real ONNX file."""
    _, _, path, X = trained
    local = LocalOnnxClient(path)
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["path"], inp = self.path, body["inputs"][0]
            seen["input"] = (inp["name"], inp["datatype"], inp["shape"])
            probs = local.predict_proba(np.array(inp["data"], dtype=np.float32).reshape(inp["shape"]))
            out = {"model_name": "bearing", "outputs": [{"name": "probabilities", "datatype": "FP32",
                                                       "shape": list(probs.shape), "data": probs.flatten().tolist()}]}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = KServeV2Client(f"http://127.0.0.1:{srv.server_port}")
        probs = client.predict_proba(X[:4])
        assert probs.shape == (4, len(CLASSES))
        assert seen["path"] == "/v2/models/bearing/infer"
        assert seen["input"] == ("features", "FP32", [4, N_FEATURES])
        np.testing.assert_allclose(probs, local.predict_proba(X[:4]), atol=1e-6)
    finally:
        srv.shutdown()
