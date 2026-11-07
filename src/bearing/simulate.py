"""Synthetic planer-motor signals: healthy, imbalance and bearing wear.

The physics is deliberately simple but recognisable to a reliability engineer:
- imbalance raises the 1x running-speed component and slightly the motor current;
- an outer-race bearing defect produces periodic impacts at the ball-pass
  frequency (BPFO) that ring a high-frequency resonance, plus a temperature rise.
Severity is random, so mild faults overlap with healthy machines and the model
has to work for its accuracy.
"""
from __future__ import annotations

import numpy as np

from .config import CLASSES, SAMPLE_RATE_HZ, WINDOW_SAMPLES

BPFO_RATIO = 3.58          # ball-pass frequency, outer race, as a multiple of shaft speed
RESONANCE_HZ = 1850.0      # structural resonance excited by bearing impacts
RESONANCE_DECAY_S = 0.004


def simulate_window(mode: str = "healthy", severity: float = 0.0, rng: np.random.Generator | None = None,
                    shaft_hz: float | None = None) -> dict:
    """One second of signal for one asset. Returns raw samples plus temperature and current."""
    if mode not in CLASSES:
        raise ValueError(f"mode must be one of {CLASSES}")
    rng = rng or np.random.default_rng()
    severity = float(np.clip(severity, 0.0, 1.0))
    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE_HZ
    f0 = shaft_hz or 30.0 * rng.uniform(0.95, 1.05)            # ~1800 rpm
    phase = rng.uniform(0, 2 * np.pi, size=3)

    a1 = rng.uniform(0.18, 0.28)                               # 1x amplitude in g
    if mode == "imbalance":
        a1 *= 1.0 + 2.5 * severity
    x = (a1 * np.sin(2 * np.pi * f0 * t + phase[0])
         + 0.3 * a1 * np.sin(2 * np.pi * 2 * f0 * t + phase[1])
         + 0.04 * np.sin(2 * np.pi * 7 * f0 * t + phase[2]))
    noise = 0.05 * (1.0 + (0.6 * severity if mode == "bearing_wear" else 0.0))
    x += rng.normal(0.0, noise, WINDOW_SAMPLES)

    temp_c = rng.normal(45.0, 2.5)
    current_a = rng.normal(18.0, 0.8)

    if mode == "imbalance":
        current_a += 1.2 * severity
    if mode == "bearing_wear":
        bpfo = BPFO_RATIO * f0
        ring_t = np.arange(int(0.03 * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
        ring = np.exp(-ring_t / RESONANCE_DECAY_S) * np.sin(2 * np.pi * RESONANCE_HZ * ring_t)
        amp = 0.15 + 1.6 * severity
        for k in range(int(bpfo * t[-1]) + 1):
            start = int((k / bpfo + rng.normal(0, 0.0004)) * SAMPLE_RATE_HZ)
            if 0 <= start < WINDOW_SAMPLES:
                end = min(WINDOW_SAMPLES, start + ring.size)
                x[start:end] += amp * rng.uniform(0.7, 1.3) * ring[: end - start]
        temp_c += 12.0 * severity

    return {"samples": x.astype(np.float32), "temp_c": float(temp_c), "current_a": float(current_a),
            "sample_rate_hz": SAMPLE_RATE_HZ}


def make_dataset(n_per_class: int = 600, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix X (n, N_FEATURES) and integer labels y, balanced across CLASSES."""
    from .features import extract_features

    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, mode in enumerate(CLASSES):
        for _ in range(n_per_class):
            sev = 0.0 if mode == "healthy" else rng.uniform(0.1, 1.0)
            w = simulate_window(mode, sev, rng)
            X.append(extract_features(w["samples"], w["temp_c"], w["current_a"]))
            y.append(label)
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
    order = rng.permutation(len(y))
    return X[order], y[order]
