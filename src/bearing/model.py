"""Train, evaluate, and export the classifier to plain ONNX.

The exported graph uses only standard ONNX operators (Sub, Div, MatMul, Add,
Relu, Softmax), so it runs unchanged in OpenVINO Model Server on OpenShift AI,
in OVMS on a MicroShift/Podman edge node, or in ONNX Runtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASSES, FEATURE_NAMES, N_FEATURES, ONNX_INPUT, ONNX_OUTPUT


def train(X: np.ndarray, y: np.ndarray, seed: int = 7, hidden=(32, 16)) -> tuple[Pipeline, dict]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=hidden, activation="relu", max_iter=2000,
                              alpha=1e-3, random_state=seed)),
    ])
    clf.fit(X_tr, y_tr)
    return clf, evaluate(clf, X_te, y_te)


def evaluate(clf: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    pred = clf.predict(X)
    return {
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "per_class_f1": {c: round(float(s), 4) for c, s in zip(CLASSES, f1_score(y, pred, average=None))},
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "report": classification_report(y, pred, target_names=CLASSES, digits=3),
        "n_test": int(len(y)),
    }


def export_onnx(clf: Pipeline, path: str | Path, opset: int = 13) -> Path:
    """Write the scaler + MLP as a standard-operator ONNX graph: features[N,10] -> probabilities[N,3]."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    scaler: StandardScaler = clf.named_steps["scale"]
    mlp: MLPClassifier = clf.named_steps["mlp"]
    if mlp.out_activation_ != "softmax":
        raise ValueError("expected a multiclass (softmax) MLP")

    inits = [numpy_helper.from_array(scaler.mean_.astype(np.float32), "mean"),
             numpy_helper.from_array(scaler.scale_.astype(np.float32), "scale")]
    nodes = [helper.make_node("Sub", [ONNX_INPUT, "mean"], ["centred"]),
             helper.make_node("Div", ["centred", "scale"], ["h0"])]
    cur = "h0"
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        inits += [numpy_helper.from_array(W.astype(np.float32), f"W{i}"),
                  numpy_helper.from_array(b.astype(np.float32), f"b{i}")]
        nodes += [helper.make_node("MatMul", [cur, f"W{i}"], [f"mm{i}"]),
                  helper.make_node("Add", [f"mm{i}", f"b{i}"], [f"z{i}"])]
        cur = f"z{i}"
        if i < len(mlp.coefs_) - 1:
            nodes.append(helper.make_node("Relu", [cur], [f"a{i}"]))
            cur = f"a{i}"
    nodes.append(helper.make_node("Softmax", [cur], [ONNX_OUTPUT], axis=1))

    graph = helper.make_graph(
        nodes, "bearing_classifier",
        [helper.make_tensor_value_info(ONNX_INPUT, TensorProto.FLOAT, ["N", N_FEATURES])],
        [helper.make_tensor_value_info(ONNX_OUTPUT, TensorProto.FLOAT, ["N", len(CLASSES)])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                              producer_name="mill-bearing-ai")
    model.ir_version = 8  # readable by older runtimes, including OVMS
    model.metadata_props.extend([
        onnx.StringStringEntryProto(key="classes", value=json.dumps(CLASSES)),
        onnx.StringStringEntryProto(key="features", value=json.dumps(FEATURE_NAMES)),
    ])
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


def verify_onnx(path: str | Path, clf: Pipeline, X: np.ndarray, atol: float = 1e-4) -> float:
    """Max absolute difference between ONNX Runtime and scikit-learn probabilities."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_p = sess.run([ONNX_OUTPUT], {ONNX_INPUT: X.astype(np.float32)})[0]
    diff = float(np.max(np.abs(onnx_p - clf.predict_proba(X))))
    if diff > atol:
        raise AssertionError(f"ONNX output differs from scikit-learn by {diff}")
    return diff
