"""S3 helpers. Reads the same environment variables an OpenShift AI connection injects
into a workbench or pipeline step: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_S3_ENDPOINT, AWS_DEFAULT_REGION, AWS_S3_BUCKET."""
from __future__ import annotations

import os
from pathlib import Path


def s3_client():
    import boto3
    from botocore.config import Config

    endpoint = os.environ.get("AWS_S3_ENDPOINT") or None
    verify = os.environ.get("AWS_S3_VERIFY", "true").lower() != "false"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        verify=verify,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def upload_model(local_onnx: str | Path, version: str = "1", prefix: str = "models/bearing",
                 bucket: str | None = None) -> str:
    """Upload to s3://bucket/<prefix>/<version>/model.onnx, the versioned layout OVMS expects.
    Returns the s3:// URI of the version folder."""
    bucket = bucket or os.environ["AWS_S3_BUCKET"]
    key = f"{prefix.strip('/')}/{version}/model.onnx"
    s3_client().upload_file(str(local_onnx), bucket, key)
    return f"s3://{bucket}/{prefix.strip('/')}/"


def ensure_bucket(bucket: str) -> None:
    client = s3_client()
    names = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    if bucket not in names:
        client.create_bucket(Bucket=bucket)
