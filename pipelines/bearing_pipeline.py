"""Data Science Pipeline (Kubeflow Pipelines v2): generate data -> train -> evaluation gate -> ONNX -> S3 -> registry.

Compile:   pip install "kfp>=2.9" kfp-kubernetes && python pipelines/bearing_pipeline.py
Then import pipelines/bearing_pipeline.yaml in the OpenShift AI dashboard (Pipelines > Import pipeline).

Every step runs in the project's own image (repo Containerfile), so the pipeline uses exactly the
same simulation, feature and export code as the notebooks and the edge scorer.
"""
from kfp import compiler, dsl, kubernetes
from kfp.dsl import Dataset, Input, Markdown, Metrics, Model, Output

# Build and push the repo Containerfile, then set this (or override at compile time with PIPELINE_IMAGE).
import os
IMAGE = os.environ.get("PIPELINE_IMAGE", "quay.io/YOUR_ORG/mill-bearing-ai:0.1.0")
S3_SECRET = "aws-connection-my-storage"   # the "My Storage" connection
S3_ENV = {k: k for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
                         "AWS_S3_ENDPOINT", "AWS_S3_BUCKET"]}


@dsl.component(base_image=IMAGE)
def generate_data(n_per_class: int, seed: int, dataset: Output[Dataset]):
    import numpy as np
    from bearing.simulate import make_dataset

    X, y = make_dataset(n_per_class, seed)
    with open(dataset.path, "wb") as f:
        np.savez(f, X=X, y=y)
    dataset.metadata.update({"rows": int(len(y)), "seed": seed})


@dsl.component(base_image=IMAGE)
def train_model(dataset: Input[Dataset], seed: int, model: Output[Model], metrics: Output[Metrics],
                report: Output[Markdown]) -> float:
    import pickle
    import numpy as np
    from bearing.model import train

    d = np.load(dataset.path)
    clf, m = train(d["X"], d["y"], seed)
    with open(model.path, "wb") as f:
        pickle.dump(clf, f)
    metrics.log_metric("macro_f1", m["macro_f1"])
    for cls, score in m["per_class_f1"].items():
        metrics.log_metric(f"f1_{cls}", score)
    with open(report.path, "w") as f:
        f.write("## Evaluation\n\n```\n" + m["report"] + "\n```\n")
    return float(m["macro_f1"])


@dsl.component(base_image=IMAGE)
def export_onnx_model(model: Input[Model], dataset: Input[Dataset], onnx_model: Output[Model]):
    import pickle
    import numpy as np
    from bearing.model import export_onnx, verify_onnx

    with open(model.path, "rb") as f:
        clf = pickle.load(f)
    path = export_onnx(clf, onnx_model.path)
    diff = verify_onnx(path, clf, np.load(dataset.path)["X"][:256])
    onnx_model.metadata.update({"framework": "onnx", "max_diff_vs_sklearn": diff})


@dsl.component(base_image=IMAGE)
def publish_model(onnx_model: Input[Model], version: str, macro_f1: float, register: bool) -> str:
    from bearing.storage import upload_model

    uri = upload_model(onnx_model.path, version)
    print(f"uploaded model version {version} to {uri}")
    if register:
        from bearing.registry import register as reg
        reg(uri, version, {"macro_f1": macro_f1})
    return uri


@dsl.pipeline(name="mill-bearing-training",
              description="Train, gate, export and publish the planer bearing fault classifier")
def bearing_pipeline(model_version: str = "2", n_per_class: int = 600, seed: int = 11,
                     min_macro_f1: float = 0.90, register_model: bool = False):
    data = generate_data(n_per_class=n_per_class, seed=seed)
    trained = train_model(dataset=data.outputs["dataset"], seed=seed)

    with dsl.If(trained.outputs["Output"] >= min_macro_f1, name="evaluation-gate"):
        exported = export_onnx_model(model=trained.outputs["model"], dataset=data.outputs["dataset"])
        published = publish_model(onnx_model=exported.outputs["onnx_model"], version=model_version,
                                  macro_f1=trained.outputs["Output"], register=register_model)
        kubernetes.use_secret_as_env(published, secret_name=S3_SECRET, secret_key_to_env=S3_ENV)

    for t in (data, trained):
        t.set_caching_options(False)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bearing_pipeline.yaml")
    compiler.Compiler().compile(bearing_pipeline, out)
    print("wrote", out)
