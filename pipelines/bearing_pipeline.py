"""Data Science Pipeline (Kubeflow Pipelines v2): generate data -> train -> evaluation gate -> ONNX -> S3 -> registry,
and in parallel ONNX -> model image ("modelcar") -> Sigstore signature, which is what the edge runs.

Compile:   pip install "kfp>=2.9" kfp-kubernetes && python pipelines/bearing_pipeline.py
Then import pipelines/bearing_pipeline.yaml in the OpenShift AI dashboard (Pipelines > Import pipeline).

Every step runs in the project's own image (images/mill-bearing-ai/Containerfile), so the pipeline uses exactly the
same simulation, feature and export code as the notebooks and the edge scorer. The modelcar steps use pinned crane
and cosign images instead: crane adds the model as one layer on the base image and pushes it with no build daemon
(so no privileged pod), and cosign signs the pushed digest with the project's key.

The modelcar steps need two secrets in the project:
  oc create secret generic modelcar-registry --type=kubernetes.io/dockerconfigjson \
      --from-file=.dockerconfigjson=$HOME/.config/containers/auth.json     # push access to MODELCAR_REPO
  cosign generate-key-pair k8s://<your-data-science-project>/cosign-signing-key
"""
from kfp import compiler, dsl, kubernetes
from kfp.dsl import Dataset, Input, Markdown, Metrics, Model, Output

# Build and push images/mill-bearing-ai/Containerfile, then set this (or override at compile time with PIPELINE_IMAGE).
import os
IMAGE = os.environ.get("PIPELINE_IMAGE", "quay.io/YOUR_ORG/mill-bearing-ai:0.1.0")
S3_SECRET = "aws-connection-my-storage"   # the "My Storage" connection
S3_ENV = {k: k for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
                         "AWS_S3_ENDPOINT", "AWS_S3_BUCKET"]}

# Model image ("modelcar"): same layout as images/bearing-model/Containerfile, tagged with the model version.
MODELCAR_REPO = os.environ.get("MODELCAR_REPO", "quay.io/YOUR_ORG/bearing-model")
MODELCAR_BASE = "registry.access.redhat.com/ubi9/ubi-minimal:latest"
REGISTRY_SECRET = "modelcar-registry"     # dockerconfigjson with push access to MODELCAR_REPO
COSIGN_SECRET = "cosign-signing-key"      # cosign.key, cosign.pub, cosign.password
CRANE_IMAGE = "gcr.io/go-containerregistry/crane/debug:v0.21.5"
# v2 writes the classic .sig signatures that Podman's sigstoreSigned policy verifies on the edge devices.
COSIGN_IMAGE = "ghcr.io/sigstore/cosign/cosign:v2.5.0-dev"


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


@dsl.container_component
def build_modelcar(onnx_model: Input[Model], repository: str, version: str, base_image: str,
                   image: dsl.OutputPath(str)):
    """Push <repository>:<version> with the model at /models/bearing/<version>/model.onnx; output the digest ref."""
    return dsl.ContainerSpec(
        image=CRANE_IMAGE,
        command=["/busybox/sh", "-ec", """
mkdir -p /tmp/docker "/tmp/layer/models/bearing/$3" "$(dirname "$5")"
cp /secrets/registry/.dockerconfigjson /tmp/docker/config.json
export DOCKER_CONFIG=/tmp/docker
cp "$1" "/tmp/layer/models/bearing/$3/model.onnx"
chmod -R a+rX /tmp/layer
tar -C /tmp/layer -cf /tmp/model-layer.tar models
crane mutate --platform linux/amd64 "$4" --append /tmp/model-layer.tar -t "$2:$3" --user 1001 \
    -l org.opencontainers.image.title=bearing-model -l org.opencontainers.image.version="$3"
printf '%s@%s' "$2" "$(crane digest "$2:$3")" > "$5"
echo "pushed $(cat "$5")"
""", "build-modelcar"],
        args=[onnx_model.path, repository, version, base_image, image])


@dsl.container_component
def sign_modelcar(image: str, version: str, macro_f1: float, tlog_upload: bool):
    """Sign the pushed digest with the project's cosign key, recording the version and evaluation score."""
    return dsl.ContainerSpec(
        image=COSIGN_IMAGE,
        command=["/busybox/sh", "-ec", """
mkdir -p /tmp/docker
cp /secrets/registry/.dockerconfigjson /tmp/docker/config.json
export DOCKER_CONFIG=/tmp/docker
case "$4" in true|True) skip_tlog=false ;; *) skip_tlog=true ;; esac
cosign sign --yes --key /secrets/cosign/cosign.key --tlog-upload="$4" \
    -a model_version="$2" -a macro_f1="$3" "$1"
cosign verify --key /secrets/cosign/cosign.pub --insecure-ignore-tlog="$skip_tlog" "$1" > /dev/null
echo "signed and verified $1"
""", "sign-modelcar"],
        args=[image, version, macro_f1, tlog_upload])


@dsl.pipeline(name="mill-bearing-training",
              description="Train, gate, export and publish the planer bearing fault classifier")
def bearing_pipeline(model_version: str = "2", n_per_class: int = 600, seed: int = 11,
                     min_macro_f1: float = 0.90, register_model: bool = False,
                     modelcar_repository: str = MODELCAR_REPO, transparency_log: bool = True):
    data = generate_data(n_per_class=n_per_class, seed=seed)
    trained = train_model(dataset=data.outputs["dataset"], seed=seed)

    with dsl.If(trained.outputs["Output"] >= min_macro_f1, name="evaluation-gate"):
        exported = export_onnx_model(model=trained.outputs["model"], dataset=data.outputs["dataset"])
        published = publish_model(onnx_model=exported.outputs["onnx_model"], version=model_version,
                                  macro_f1=trained.outputs["Output"], register=register_model)
        kubernetes.use_secret_as_env(published, secret_name=S3_SECRET, secret_key_to_env=S3_ENV)

        modelcar = build_modelcar(onnx_model=exported.outputs["onnx_model"], repository=modelcar_repository,
                                  version=model_version, base_image=MODELCAR_BASE)
        signed = sign_modelcar(image=modelcar.outputs["image"], version=model_version,
                               macro_f1=trained.outputs["Output"], tlog_upload=transparency_log)
        for t in (modelcar, signed):
            kubernetes.use_secret_as_volume(t, secret_name=REGISTRY_SECRET, mount_path="/secrets/registry")
        kubernetes.use_secret_as_volume(signed, secret_name=COSIGN_SECRET, mount_path="/secrets/cosign")
        kubernetes.use_secret_as_env(signed, secret_name=COSIGN_SECRET,
                                     secret_key_to_env={"cosign.password": "COSIGN_PASSWORD"})

    for t in (data, trained):
        t.set_caching_options(False)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bearing_pipeline.yaml")
    compiler.Compiler().compile(bearing_pipeline, out)
    print("wrote", out)
