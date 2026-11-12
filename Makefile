REGISTRY ?= quay.io/rprakashg
APP_IMAGE ?= $(REGISTRY)/mill-bearing-ai:0.1.0
MODEL_IMAGE ?= $(REGISTRY)/bearing-model:1
EDGE_DEVICE_IMAGE ?= $(REGISTRY)/mill-edge-device:0.1.0
EDGE_DEVICE_WITH_MICROSHIFT_IMAGE ?= $(REGISTRY)/mill-edge-microshift-device:0.1.0
PY ?= python3

.PHONY: install
install:            ## install the package + dev tools into the current Python environment
	$(PY) -m pip install -e ".[notebook,dev]"

.PHONY: test
test:               ## unit tests: features, ONNX parity, scorer debounce/fallback, KServe v2 client
	$(PY) -m pytest -q tests

.PHONY: train
train:              ## train and export models/bearing/1/model.onnx for the laptop edge
	$(PY) -m bearing.train --out models --version 1

.PHONY: notebooks
notebooks:          ## regenerate the workbench notebooks from notebooks/_build_notebooks.py
	$(PY) notebooks/_build_notebooks.py

.PHONY: pipeline
pipeline:           ## compile the Data Science Pipeline (needs kfp + kfp-kubernetes)
	PIPELINE_IMAGE=$(APP_IMAGE) $(PY) pipelines/bearing_pipeline.py

.PHONY: image
image:              ## app image: simulator, scorer and pipeline steps
	podman build --platform=linux/amd64 -f images/mill-bearing-ai/Containerfile -t $(APP_IMAGE) .

.PHONY: model-image
model-image: train  ## "modelcar" image carrying the ONNX model for the edge
	podman build --platform=linux/amd64 -f images/bearing-model/Containerfile -t $(MODEL_IMAGE) .

.PHONY: edge-device-image
edge-device-image:
	podman build --platform=linux/amd64 -f images/edge-device/Containerfile -t $(EDGE_DEVICE_IMAGE)	. && \
	podman build --platform=linux/amd64 -f images/edge-device-with-microshift/Containerfile -t $(EDGE_DEVICE_WITH_MICROSHIFT_IMAGE) .

.PHONY: push	
push: image model-image
	podman push $(APP_IMAGE) \
	&& podman push $(MODEL_IMAGE) \
	&& podman push $(EDGE_DEVICE_IMAGE) \
	&& podman push $(EDGE_DEVICE_WITH_MICROSHIFT_IMAGE)

.PHONY: edge-up
edge-up: train      ## laptop mill edge: MQTT + OVMS + scorer + simulator
	podman compose -f edge/compose.yaml up --build

.PHONY: edge-down
edge-down:
	podman compose -f edge/compose.yaml down

.PHONY: fault
fault:              ## inject bearing wear into the running simulator
	podman exec mill-edge-scorer python -m bearing.edge.inject bearing_wear 0.6 || \
	  MQTT_HOST=localhost $(PY) -m bearing.edge.inject bearing_wear 0.6

.PHONY: heal
heal:               ## back to healthy
	podman exec mill-edge-scorer python -m bearing.edge.inject healthy || \
	  MQTT_HOST=localhost $(PY) -m bearing.edge.inject healthy

.PHONY: model-down
model-down:         ## stop the model server: the scorer keeps alerting with engineering rules
	podman stop mill-edge-ovms

.PHONY: model-up
model-up:
	podman start mill-edge-ovms
