# Customer walkthrough: mill bearing failure prediction

A spoken narrative for presenting this demo to a customer. About 15–20 minutes plus Q&A.
**Say** lines are suggested wording. Put them in your own words once you know the flow.
**Show** lines are what should be on screen.

---

## Before the meeting (checklist)

- [ ] `make test` passes on your laptop.
- [ ] `make edge-up` is running, and scorer logs show `healthy` lines in a visible terminal.
- [ ] A second terminal is ready for `make fault`, `make heal`, `make model-down` and `make model-up`.
- [ ] Notebooks 1–3 have been run once in the workbench, so plots and outputs are already rendered.
- [ ] The model is deployed in OpenShift AI as `bearing`, and the endpoint URL works in notebook 3.
- [ ] A completed pipeline run is open in the dashboard, showing the graph with the evaluation gate.
- [ ] `docs/architecture.png` is open in a tab.
- [ ] Check your own confusion matrix. The script below assumes the only errors are mild imbalance called healthy. If your run differs, change that line.

**Know your audience.** For operations or maintenance leaders, spend more time on Acts 1 and 6 and go quickly through 3–5. For data science or platform teams, slow down on Acts 3–5 and the Q&A.

---

## Act 0: Opening (1 min)

**Show:** the architecture diagram.

**Say:**
> "I want to show you one model on its whole journey, from a data scientist's notebook to a motor on the mill floor. The use case is a planer motor in a sawmill. When a bearing fails there, you lose the line for hours and sometimes the motor too. We'll detect two early faults, imbalance and bearing wear, and do it at the edge in milliseconds, even when the network is down."

> "Everything you'll see runs on Red Hat OpenShift AI centrally and on a small edge footprint at the mill. It's the same model file in both places."

**Transition:** "Let's start with what the problem looks like in the data."

---

## Act 1: The problem (2 min)

**Show:** notebook 1, the plot of the three waveforms (healthy, imbalance, bearing wear).

**Say:**
> "This is one second of vibration from an accelerometer on the motor, sampled about five thousand times a second."

> "Healthy is a clean wave at running speed, about 1,800 rpm. Imbalance makes that same wave bigger, because the rotor is heavy on one side. Bearing wear is different: every time a ball rolls over a defect in the outer race, there's a small impact. That impact makes the housing ring at a high frequency, so you see these sharp periodic spikes."

> "Severe faults are easy to see. The value is in the mild ones, which look almost healthy. Catching those early is what turns an unplanned outage into a planned work order."

**If asked "is this real data?":**
> "It's simulated with standard reliability physics, the ball-pass frequency and resonance, so we can control faults on stage. Swapping in a real sensor or an OPC UA feed only replaces the data source. Feature extraction, the model and serving stay the same."

**Transition:** "So how do we turn that signal into something a model can learn from?"

---

## Act 2: Features and training (2 min)

**Show:** the feature table, then the confusion matrix in notebook 1.

**Say:**
> "We reduce each second of signal to ten numbers a reliability engineer would recognize: overall vibration level, peak, crest factor, kurtosis (how spiky the signal is), energy in four frequency bands, plus bearing temperature and motor current."

> "One important detail: this feature code lives in one shared Python package. Training, the pipeline and the edge scorer all call the same function, so what the edge sees is computed exactly the way the model was trained on. That removes a very common source of production bugs."

> "The model itself is small, a scaler plus a small neural network. Look at the confusion matrix: the only mistakes are mild imbalance being called healthy. Where to draw that line is a business decision you'd make with maintenance. It isn't a data science decision."

**Transition:** "A trained model in a notebook is worth nothing until it runs somewhere. Let's make it portable."

---

## Act 3: A portable model (1–2 min)

**Show:** notebook 2, the cell that lists the ONNX operators, then the bucket path `models/bearing/1/model.onnx`.

**Say:**
> "We export to ONNX, an open model format. The whole model comes down to six standard operations: subtract, divide, matrix multiply, add, ReLU and softmax. We also check automatically that the ONNX version gives the same answers as the original, to about seven decimal places."

> "Because it's standard ONNX, there's no lock-in on the serving side. The same file runs in the OpenVINO model server on OpenShift AI, at the edge, or in ONNX Runtime on a laptop."

> "We store it in S3-compatible storage in a versioned folder layout. Optionally we register it in the Model Registry with its metrics, so there's a record of every version and how good it was."

**Transition:** "Now let's serve it."

---

## Act 4: Serving (1–2 min)

**Show:** the deployed model `bearing` in the OpenShift AI dashboard, then run notebook 3.

**Say:**
> "This is deployed with KServe and the OpenVINO Model Server, a few clicks in the dashboard or one YAML file. It exposes the standard KServe v2 inference API. Any application that speaks that protocol can call it, and the edge uses the exact same API."

> "Each call takes about a millisecond. Healthy, imbalance and bearing wear samples come back with the right labels and confidence scores."

**Transition:** "That's fine for a one-off, but you don't want a person running notebooks every time there's new data."

---

## Act 5: Automation with a quality gate (1–2 min)

**Show:** the pipeline run graph in *Data Science Pipelines*, with the `evaluation-gate` step highlighted.

**Say:**
> "This pipeline does the same steps automatically: generate or pull data, train, evaluate, export and publish. Note the evaluation gate: if macro F1 is below 0.90, the pipeline stops and nothing gets published. No model reaches production without passing evaluation."

> "Every step runs in the same container image, using the same code as the notebooks and the edge. When it passes, it publishes a new version, and that version is recorded with its metrics."

**Transition:** "Now the part that matters most to the people running the mill: what happens on the floor."

---

## Act 6: The mill edge (3–5 min) — the main moment

**Show:** the terminal with scorer logs scrolling `planer-3 healthy conf=0.9x … ms [model]`.

**Say:**
> "This is the edge. A simulated motor publishes vibration once a second over MQTT, the usual protocol on plant floors. The scorer computes the features locally, calls the model server running locally, and publishes alerts. Nothing here goes back to the data center to make a decision."

### 6a. Inject a fault

**Do:** `make fault` (bearing wear, severity 0.6).

**Say (while waiting):**
> "I just put bearing wear into the motor. Watch the labels change."

> "Notice it doesn't alert on the first bad reading. It waits for three abnormal windows in a row, about three seconds. That debounce is deliberate: maintenance teams stop trusting a system that cries wolf."

**Show:** the `ALERT` line.

> "There's the alert: bearing wear on planer-3, high severity, with a recommended action to create a work order and inspect within 48 hours. It also carries the feature values, so a reliability engineer can see *why*."

### 6b. Heal

**Do:** `make heal`.

> "Once the motor is healthy again, the alert clears automatically. There's no stale alarm left for someone to acknowledge."

### 6c. Lose the model server

**Do:** `make model-down`, then `make fault` again.

**Say:**
> "Now the worst case: the model server crashes, or the WAN drops during an update. Most AI demos go blind here."

**Show:** log lines tagged `[rules (URLError)]` or similar, then the alert.

> "The scorer notices immediately and switches to engineering rules, based on thresholds for kurtosis and high-frequency energy. It's less precise, but it keeps alerting. It degrades instead of going blind. In a safety- or uptime-critical environment, that's the difference between a pilot and something you can actually run."

**Do:** `make model-up`, then `make heal`.

> "The model is back, and the scorer is using it again with no restart."

---

## Act 7: Scaling to every mill (1–2 min)

**Show:** the architecture diagram again, pointing at the edge side.

**Say:**
> "At a real mill this runs on MicroShift, a small Kubernetes for edge devices, with the same manifests. The model ships as a container image, a 'modelcar'. A new model version is just a new image tag in Git."

> "That means models go through the same supply chain you already trust for software: sign them with cosign so the edge only runs approved models, and roll them out with OpenShift GitOps or Red Hat Edge Manager, one mill or a hundred."

---

## Close (1 min)

**Say:**
> "To recap: one platform and one model file, from the notebook to the mill floor. You saw a quality gate so no bad model ships, a registry so every version is traceable, signed images and GitOps so the edge only runs what you approved, and an edge that keeps working when the network doesn't."

> "That governance is what lets you go from one pilot to every mill."

**Ask:**
> "Which assets cost you the most when they go down today? And what does your sensor data look like: historians, OPC UA, something else? That tells us where a pilot would start."

---

## Likely questions and suggested answers

**"Can we use our own data?"**
> Yes. Replace the simulator with your historian or OPC UA feed. Feature extraction, the model, serving, the pipeline and the edge stay the same.

**"How much labeled failure data do we need?"**
> Real failures are rare, so that's normal. Common approaches: start with the engineering rules (already in the scorer), use maintenance records to label past windows, add physics-based simulation like this demo, or start with anomaly detection and add classification as labels build up.

**"Why not just use thresholds?"**
> Thresholds are the fallback here, and they work for severe faults. The model catches mild, early faults that thresholds miss or misclassify, and that's where planning time comes from.

**"What about GPUs?"**
> This model doesn't need one. It runs on CPU in about a millisecond. OpenShift AI supports GPUs through hardware profiles and Kueue queues for larger models such as vision or deep learning on raw signals.

**"What happens when the data drifts, like new motors or seasons?"**
> Compare live edge feature distributions with the training data. When they diverge, trigger the pipeline. The gate keeps a retrained model from shipping unless it's at least as good as the bar you set.

**"Are we locked into Red Hat for serving?"**
> No. It's plain ONNX and the standard KServe v2 protocol, so it runs anywhere those are supported.

**"What about latency and network?"**
> Decisions happen locally at the edge, so WAN latency doesn't matter. The central platform is for training, governance and rollout.

**"Is this production ready?"**
> It's a reference demo. The patterns (gate, registry, signed images, GitOps, fallback) are production patterns. A real deployment adds your data, alert routing to your CMMS or work-order system, monitoring, and validation with your maintenance team.

---

## If something goes wrong live

- **The alert doesn't fire:** remind them it needs three windows in a row (about 3 s). If it still doesn't fire, run `make fault` again. Severity 0.6 is reliable, and 0.8 is more dramatic.
- **The model endpoint fails in notebook 3:** notebook 3 falls back to local ONNX Runtime when `MODEL_URL` isn't set. Say "same model, same answers, running locally" and continue.
- **The edge stack won't start:** present Act 6 from the code (`src/bearing/edge/scorer.py`, `Scorer.handle`) and walk through the debounce and fallback logic.
- **The cluster is unavailable:** Path A (laptop) covers training, ONNX, OVMS serving and the whole edge story by itself.

---

## Practice tips

1. Rehearse Act 6 the most. It's the moment people remember, and timing the alert while you talk takes practice.
2. Time yourself. If you go over 20 minutes, cut Act 3 and Act 5 down to one sentence each.
3. Learn three numbers: **10 features**, **3 windows to alert**, **0.90 F1 gate**. Add **~1 ms per call** if your environment shows it.
4. Practice the fallback line without notes: *"It degrades, it doesn't go blind."*
