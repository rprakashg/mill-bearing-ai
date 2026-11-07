# One image for the edge apps (simulator, scorer) and the pipeline steps.
FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /opt/app-root/src/mill-bearing-ai
COPY --chown=1001:0 pyproject.toml ./
COPY --chown=1001:0 src ./src
RUN pip install --no-cache-dir . && python -c "import bearing, onnxruntime, paho.mqtt"

USER 1001
CMD ["python", "-m", "bearing.edge.scorer"]
