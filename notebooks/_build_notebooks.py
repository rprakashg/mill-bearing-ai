"""Generates the three workbench notebooks (kept as code so they stay reviewable). Run: python notebooks/_build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = code("""# Install the demo package from the cloned repo (skipped if already installed)
import importlib.util
if importlib.util.find_spec("bearing") is None:
    %pip install -q -e "..[notebook]"
    print("Installed. If imports fail, restart the kernel once.")""")

nb1 = [
    md("""# 1. Explore the data and train the bearing fault model

**Story:** a planer motor in a sawmill. We want to spot **imbalance** and **bearing wear** early, from one second of vibration plus temperature and motor current.

This notebook: simulate sensor data, look at what faults look like, compute features, train a small neural network, and evaluate it."""),
    SETUP,
    code("""import numpy as np
import matplotlib.pyplot as plt
from bearing.config import CLASSES, FEATURE_NAMES, SAMPLE_RATE_HZ
from bearing.simulate import simulate_window, make_dataset
from bearing.features import extract_features, as_dict"""),
    md("## What does a failing bearing look like?\nSame motor, three conditions. Bearing wear shows up as sharp periodic impacts that ring a high-frequency resonance."),
    code("""rng = np.random.default_rng(1)
fig, axes = plt.subplots(3, 2, figsize=(12, 7))
for row, (mode, sev) in enumerate([("healthy", 0), ("imbalance", 0.8), ("bearing_wear", 0.8)]):
    w = simulate_window(mode, sev, rng, shaft_hz=30)
    t = np.arange(w["samples"].size) / SAMPLE_RATE_HZ
    axes[row, 0].plot(t[:1024], w["samples"][:1024], lw=0.7)
    axes[row, 0].set_title(f"{mode}: waveform (first 0.2 s)"); axes[row, 0].set_ylabel("g")
    spec = np.abs(np.fft.rfft(w["samples"])) / w["samples"].size
    f = np.fft.rfftfreq(w["samples"].size, 1 / SAMPLE_RATE_HZ)
    axes[row, 1].semilogy(f, spec + 1e-6, lw=0.6); axes[row, 1].set_title(f"{mode}: spectrum")
    axes[row, 1].set_xlabel("Hz")
    print(mode, as_dict(extract_features(w["samples"], w["temp_c"], w["current_a"])))
plt.tight_layout(); plt.show()"""),
    md("## Build a training set\n600 one-second windows per class. Fault severity is random, so mild faults look almost healthy, which is what makes early detection hard."),
    code("""X, y = make_dataset(n_per_class=600, seed=7)
print(X.shape, "features:", FEATURE_NAMES)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i, c in enumerate(CLASSES):
    ax[0].scatter(X[y == i, 4], X[y == i, 7], s=6, label=c)
    ax[1].hist(X[y == i, 3], bins=40, alpha=0.6, label=c)
ax[0].set_xlabel("1x band (g)"); ax[0].set_ylabel("bearing resonance band (g)"); ax[0].legend()
ax[1].set_xlabel("kurtosis"); ax[1].legend(); plt.show()"""),
    md("## Train and evaluate\nA small scaler + neural network. Small on purpose: it has to run in milliseconds on a mill edge server."),
    code("""from bearing.model import train
clf, metrics = train(X, y, seed=7)
print(metrics["report"])
print("confusion matrix (rows = truth):", CLASSES)
print(np.array(metrics["confusion_matrix"]))"""),
    code("""# Save for notebook 2
import pickle, os, json
os.makedirs("../models", exist_ok=True)
with open("../models/bearing-sklearn.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("../models/bearing-metrics.json", "w") as f:
    json.dump({k: v for k, v in metrics.items() if k != "report"}, f, indent=2)
print("saved ../models/bearing-sklearn.pkl")"""),
    md("**Talking point:** the mistakes are mild imbalance called healthy. That is the business trade-off to discuss with maintenance: alert earlier and accept more false alarms, or wait for more evidence."),
]

nb2 = [
    md("""# 2. Export to ONNX, upload to S3, register

ONNX makes the model portable: the same file runs in OpenVINO Model Server on OpenShift AI and on a MicroShift edge node.
Upload uses the **My Storage** connection attached to this workbench (its `AWS_*` environment variables)."""),
    SETUP,
    code("""import os, json, pickle
from bearing.model import export_onnx, verify_onnx
from bearing.simulate import make_dataset

with open("../models/bearing-sklearn.pkl", "rb") as f:
    clf = pickle.load(f)
metrics = json.load(open("../models/bearing-metrics.json"))
VERSION = "1"
path = export_onnx(clf, f"../models/bearing/{VERSION}/model.onnx")
X_check, _ = make_dataset(50, seed=99)
print("exported", path, "| max difference vs scikit-learn:", verify_onnx(path, clf, X_check))"""),
    code("""import onnx
m = onnx.load(path)
print("inputs: ", [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print("outputs:", [(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
print("operators:", sorted({n.op_type for n in m.graph.node}))"""),
    md("## Upload to S3 (layout OVMS expects: `models/bearing/<version>/model.onnx`)"),
    code("""if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_S3_BUCKET"):
    from bearing.storage import upload_model
    MODEL_URI = upload_model(path, VERSION)
    print("uploaded:", MODEL_URI)
else:
    MODEL_URI = None
    print("No S3 connection attached to this workbench; skipping upload (attach 'My Storage' to enable).")"""),
    md("## Register version 1 in the Model Registry (optional)\nSet `MODEL_REGISTRY_URL` to your registry's REST route. Skipped if not set."),
    code("""if MODEL_URI:
    from bearing.registry import register
    register(MODEL_URI, VERSION, metrics)"""),
    md("**Next:** deploy it. Dashboard: *Models > Deploy model* with the OpenVINO Model Server runtime, framework `onnx - 1`, connection *My Storage*, path `models/bearing/`. Or `oc apply -f openshift-ai/10-serving.yaml`."),
]

nb3 = [
    md("""# 3. Call the deployed model

Sends fresh simulated windows for each condition to the model's KServe v2 (Open Inference Protocol) REST endpoint.
Set `MODEL_URL` to the inference endpoint shown in the dashboard (internal or external). If unset, the notebook scores locally with ONNX Runtime so you can still test the flow."""),
    SETUP,
    code("""import os, time
import numpy as np
from bearing.simulate import simulate_window
from bearing.features import extract_features
from bearing.inference import KServeV2Client, LocalOnnxClient, label

MODEL_URL = os.environ.get("MODEL_URL", "")   # e.g. https://bearing-<project>.apps.<cluster>
if MODEL_URL:
    client = KServeV2Client(MODEL_URL, token=os.environ.get("MODEL_TOKEN"),
                            verify_tls=os.environ.get("MODEL_VERIFY_TLS", "true") != "false")
    print("calling", client.url)
else:
    client = LocalOnnxClient("../models/bearing/1/model.onnx")
    print("MODEL_URL not set: scoring locally with ONNX Runtime")"""),
    code("""rng = np.random.default_rng(42)
for mode, sev in [("healthy", 0), ("imbalance", 0.3), ("imbalance", 0.9), ("bearing_wear", 0.2), ("bearing_wear", 0.9)]:
    w = simulate_window(mode, sev, rng)
    x = extract_features(w["samples"], w["temp_c"], w["current_a"])
    t0 = time.perf_counter(); probs = client.predict_proba(x); ms = (time.perf_counter() - t0) * 1000
    (pred, conf), = label(probs)
    print(f"truth={mode:<13} severity={sev:.1f}  ->  {pred:<13} {conf:.2f}   ({ms:.1f} ms)")"""),
    md("""## Raw request, for curl or Postman
```
POST {MODEL_URL}/v2/models/bearing/infer
{"inputs": [{"name": "features", "shape": [1, 10], "datatype": "FP32", "data": [ ...10 numbers... ]}]}
```"""),
    code("""import json
w = simulate_window("bearing_wear", 0.8, rng)
x = extract_features(w["samples"], w["temp_c"], w["current_a"])
print(json.dumps({"inputs": [{"name": "features", "shape": [1, 10], "datatype": "FP32", "data": [round(float(v), 4) for v in x]}]}))"""),
]

for name, cells in [("1_explore_and_train.ipynb", nb1), ("2_export_upload_register.ipynb", nb2), ("3_test_endpoint.ipynb", nb3)]:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / name)
    print("wrote", name)
