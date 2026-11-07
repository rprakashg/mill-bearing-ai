"""Inject a fault into the running simulator: python -m bearing.edge.inject bearing_wear 0.8"""
import json
import os
import sys

import paho.mqtt.publish as publish

from ..config import CLASSES

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CLASSES:
        sys.exit(f"usage: python -m bearing.edge.inject {{{'|'.join(CLASSES)}}} [severity 0-1] [asset]")
    msg = {"mode": sys.argv[1],
           "severity": float(sys.argv[2]) if len(sys.argv) > 2 else (0.0 if sys.argv[1] == "healthy" else 0.8)}
    if len(sys.argv) > 3:
        msg["asset"] = sys.argv[3]
    publish.single("mill/control", json.dumps(msg), hostname=os.environ.get("MQTT_HOST", "localhost"),
                   port=int(os.environ.get("MQTT_PORT", "1883")))
    print("sent", msg)
