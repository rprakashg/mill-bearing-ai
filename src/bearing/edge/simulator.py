"""Planer-motor simulator: publishes one second of vibration per second over MQTT.

Fault injection on stage, from any MQTT client:
    mosquitto_pub -t mill/control -m '{"mode":"bearing_wear","severity":0.8}'
or  python -m bearing.edge.inject bearing_wear 0.8
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

from ..config import CLASSES
from ..simulate import simulate_window


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default=os.environ.get("MQTT_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MQTT_PORT", "1883")))
    ap.add_argument("--asset", default=os.environ.get("ASSET_ID", "planer-3"))
    ap.add_argument("--line", default=os.environ.get("LINE_ID", "line-1"))
    ap.add_argument("--mode", default=os.environ.get("FAULT_MODE", "healthy"), choices=CLASSES)
    ap.add_argument("--severity", type=float, default=float(os.environ.get("FAULT_SEVERITY", "0")))
    ap.add_argument("--interval", type=float, default=float(os.environ.get("INTERVAL_S", "1.0")))
    a = ap.parse_args()

    state = {"mode": a.mode, "severity": a.severity}
    lock = threading.Lock()
    topic = f"mill/{a.line}/{a.asset}/vibration"

    def on_connect(client, userdata, flags, reason_code, properties):
        client.subscribe("mill/control")
        print(f"connected to {a.broker}:{a.port}; publishing {topic}", flush=True)

    def on_message(client, userdata, msg):
        try:
            cmd = json.loads(msg.payload)
            if cmd.get("asset", a.asset) != a.asset:
                return
            mode = cmd.get("mode", "healthy")
            if mode not in CLASSES:
                raise ValueError(mode)
            with lock:
                state.update(mode=mode, severity=float(cmd.get("severity", 0.0 if mode == "healthy" else 0.7)))
            print(f"fault injection -> {state}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ignored control message {msg.payload!r}: {e}", flush=True)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim-{a.asset}")
    client.on_connect, client.on_message = on_connect, on_message
    client.connect_async(a.broker, a.port)
    client.loop_start()

    rng = np.random.default_rng()
    while True:
        with lock:
            mode, sev = state["mode"], state["severity"]
        w = simulate_window(mode, sev, rng, shaft_hz=30.0)
        payload = {"asset": a.asset, "line": a.line, "ts": time.time(), "sample_rate_hz": w["sample_rate_hz"],
                   "temp_c": round(w["temp_c"], 2), "current_a": round(w["current_a"], 2),
                   "samples": np.round(w["samples"], 4).tolist(), "truth": mode}
        client.publish(topic, json.dumps(payload), qos=0)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
