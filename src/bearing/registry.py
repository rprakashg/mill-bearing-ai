"""Optional: register a model version in the OpenShift AI Model Registry.

Needs `pip install model-registry` plus MODEL_REGISTRY_URL (the registry's REST
route, https://...) and a token (the workbench's service-account token, or
`oc whoami -t`). Skips cleanly when not configured, so the rest of the demo works
on clusters without a registry.
"""
from __future__ import annotations

import os
from pathlib import Path


def _token() -> str | None:
    tok = os.environ.get("MODEL_REGISTRY_TOKEN")
    sa = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    if not tok and sa.exists():
        tok = sa.read_text().strip()
    return tok


def register(model_uri: str, version: str, metrics: dict, name: str = "mill-bearing-classifier",
             author: str | None = None) -> str | None:
    url = os.environ.get("MODEL_REGISTRY_URL")
    if not url:
        print("MODEL_REGISTRY_URL not set; skipping model registry step.")
        return None
    from model_registry import ModelRegistry  # pip install model-registry

    host, _, port = url.rstrip("/").rpartition(":")
    if not host.startswith("http") or not port.isdigit():
        host, port = url.rstrip("/"), "443"
    registry = ModelRegistry(server_address=host, port=int(port),
                             author=author or os.environ.get("USER", "demo"),
                             is_secure=os.environ.get("MODEL_REGISTRY_VERIFY", "true") != "false",
                             user_token=_token())
    registry.register_model(
        name,
        model_uri,
        version=version,
        description="Planer-motor bearing fault classifier (healthy / imbalance / bearing_wear)",
        model_format_name="onnx",
        model_format_version="1",
        metadata={"macro_f1": str(metrics.get("macro_f1")), "features": "10", "classes": "3"},
    )
    print(f"Registered {name} version {version} -> {model_uri}")
    return version
