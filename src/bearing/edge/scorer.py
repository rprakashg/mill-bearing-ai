"""Edge scorer: MQTT vibration in, local inference, debounced alerts out.

Flow per message:
  raw samples -> extract_features (same code as training) -> model server (KServe v2)
  -> label + confidence -> alert after N consecutive abnormal windows -> mill/alerts/<asset>
If the model server is unreachable, simple engineering rules keep alerting (degraded mode).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import paho.mqtt.client as mqtt

from ..config import CLASSES, FEATURE_NAMES
from ..features import as_dict, extract_features
from ..inference import KServeV2Client, LocalOnnxClient

KURTOSIS_LIMIT = 3.0          # rule fallback: impacts make the signal spiky
HF_BAND_LIMIT_G = 0.12        # rule fallback: energy in the bearing resonance band
IMBALANCE_1X_LIMIT_G = 0.30   # rule fallback: high 1x running-speed vibration


def rule_based(features) -> tuple[str, float]:
    f = dict(zip(FEATURE_NAMES, features))
    if f["kurtosis"] > KURTOSIS_LIMIT or f["band_1500_2560_g"] > HF_BAND_LIMIT_G:
        return "bearing_wear", 0.0
    if f["band_0_100_g"] > IMBALANCE_1X_LIMIT_G:
        return "imbalance", 0.0
    return "healthy", 0.0


class Scorer:
    def __init__(self, model, threshold: float = 0.7, consecutive: int = 3, publish=None):
        self.model, self.threshold, self.consecutive = model, threshold, consecutive
        self.publish = publish or (lambda topic, payload: print(topic, payload, flush=True))
        self.streak = defaultdict(int)
        self.active = {}

    def handle(self, msg: dict) -> dict:
        t0 = time.perf_counter()
        feats = extract_features(msg["samples"], msg["temp_c"], msg["current_a"], msg.get("sample_rate_hz", 5120))
        source = "model"
        try:
            probs = self.model.predict_proba(feats)[0]
            idx = int(probs.argmax())
            label, conf = CLASSES[idx], float(probs[idx])
        except Exception as e:  # noqa: BLE001  model server down: degrade, don't go blind
            label, conf = rule_based(feats)
            source = f"rules ({type(e).__name__})"
        latency_ms = (time.perf_counter() - t0) * 1000
        asset = msg.get("asset", "unknown")
        abnormal = label != "healthy" and (conf >= self.threshold or source != "model")
        self.streak[asset] = self.streak[asset] + 1 if abnormal else 0

        result = {"asset": asset, "ts": msg.get("ts", time.time()), "label": label, "confidence": round(conf, 3),
                  "source": source, "latency_ms": round(latency_ms, 1), "truth": msg.get("truth")}
        if self.streak[asset] >= self.consecutive and self.active.get(asset) != label:
            self.active[asset] = label
            self.publish(f"mill/alerts/{asset}", json.dumps({
                **result, "severity": "high" if label == "bearing_wear" else "medium",
                "message": f"{label.replace('_', ' ')} detected on {asset}",
                "recommended_action": "Create maintenance work order; inspect bearing within 48 h"
                if label == "bearing_wear" else "Schedule balancing check",
                "features": as_dict(feats)}))
        elif self.streak[asset] == 0 and asset in self.active:
            del self.active[asset]
            self.publish(f"mill/alerts/{asset}", json.dumps({**result, "severity": "cleared",
                                                             "message": f"{asset} back to healthy"}))
        return result


def build_model(args):
    if args.onnx:
        return LocalOnnxClient(args.onnx)
    return KServeV2Client(args.model_url, token=os.environ.get("MODEL_TOKEN"),
                          verify_tls=os.environ.get("MODEL_VERIFY_TLS", "true") != "false")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default=os.environ.get("MQTT_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MQTT_PORT", "1883")))
    ap.add_argument("--model-url", default=os.environ.get("MODEL_URL", "http://localhost:8000"))
    ap.add_argument("--onnx", default=os.environ.get("MODEL_ONNX_PATH"), help="score locally with ONNX Runtime instead")
    ap.add_argument("--threshold", type=float, default=float(os.environ.get("ALERT_THRESHOLD", "0.7")))
    ap.add_argument("--consecutive", type=int, default=int(os.environ.get("ALERT_CONSECUTIVE", "3")))
    a = ap.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edge-scorer")
    scorer = Scorer(build_model(a), a.threshold, a.consecutive,
                    publish=lambda t, p: (client.publish(t, p, qos=1), print("ALERT", p, flush=True)))

    def on_connect(c, userdata, flags, reason_code, properties):
        c.subscribe("mill/+/+/vibration")
        print(f"scorer connected; model={'onnx:' + a.onnx if a.onnx else a.model_url}", flush=True)

    def on_message(c, userdata, msg):
        r = scorer.handle(json.loads(msg.payload))
        print(f"{r['asset']:<10} {r['label']:<13} conf={r['confidence']:.2f} {r['latency_ms']:>6.1f} ms  "
              f"[{r['source']}] truth={r['truth']}", flush=True)

    client.on_connect, client.on_message = on_connect, on_message
    client.connect(a.broker, a.port)
    client.loop_forever()


if __name__ == "__main__":
    main()
