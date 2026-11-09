REGISTRY ?= quay.io/rprakashg
APP_IMAGE ?= $(REGISTRY)/mill-bearing-ai:0.1.0
MODEL_IMAGE ?= $(REGISTRY)/bearing-model:1
PY ?= python3

.PHONY: install test train notebooks pipeline image model-image push edge-up edge-down fault heal model-down model-up

install:            ## install the package + dev tools into the current Python environment
	$(PY) -m pip install -e ".[notebook,dev]"

test:               ## unit tests: features, ONNX parity, scorer debounce/fallback, KServe v2 client
	$(PY) -m pytest -q tests

train:              ## train and export models/bearing/1/model.onnx for the laptop edge
	$(PY) -m bearing.train --out models --version 1

notebooks:          ## regenerate the workbench notebooks from notebooks/_build_notebooks.py
	$(PY) notebooks/_build_notebooks.py

pipeline:           ## compile the Data Science Pipeline (needs kfp + kfp-kubernetes)
	PIPELINE_IMAGE=$(APP_IMAGE) $(PY) pipelines/bearing_pipeline.py

image:              ## app image: simulator, scorer and pipeline steps
	podman build -t $(APP_IMAGE) -t localhost/mill-bearing-ai:latest .

model-image: train  ## "modelcar" image carrying the ONNX model for the edge
	podman build -f edge/Containerfile.model -t $(MODEL_IMAGE) .

push: image model-image
	podman push $(APP_IMAGE) && podman push $(MODEL_IMAGE)

edge-up: train      ## laptop mill edge: MQTT + OVMS + scorer + simulator
	podman compose -f edge/compose.yaml up --build

edge-down:
	podman compose -f edge/compose.yaml down

fault:              ## inject bearing wear into the running simulator
	podman exec mill-edge-scorer python -m bearing.edge.inject bearing_wear 0.6 || \
	  MQTT_HOST=localhost $(PY) -m bearing.edge.inject bearing_wear 0.6

heal:               ## back to healthy
	podman exec mill-edge-scorer python -m bearing.edge.inject healthy || \
	  MQTT_HOST=localhost $(PY) -m bearing.edge.inject healthy

model-down:         ## stop the model server: the scorer keeps alerting with engineering rules
	podman stop mill-edge-ovms

model-up:
	podman start mill-edge-ovms
