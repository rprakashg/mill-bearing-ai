"""Feature extraction: the single source of truth for training AND the edge scorer."""
from __future__ import annotations

import numpy as np

from .config import FEATURE_NAMES, SAMPLE_RATE_HZ

BANDS_HZ = [(0, 100), (100, 500), (500, 1500), (1500, SAMPLE_RATE_HZ / 2)]


def extract_features(samples, temp_c: float, current_a: float, sample_rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    crest = peak / rms if rms > 0 else 0.0
    kurt = float(np.mean(x ** 4) / (np.mean(x ** 2) ** 2)) if rms > 0 else 0.0

    spec = np.abs(np.fft.rfft(x)) / x.size
    freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate_hz)
    bands = []
    for lo, hi in BANDS_HZ:
        m = (freqs >= lo) & (freqs < hi)
        bands.append(float(np.sqrt(2.0 * np.sum(spec[m] ** 2))))  # RMS amplitude in band

    vec = np.array([rms, peak, crest, kurt, *bands, float(temp_c), float(current_a)], dtype=np.float32)
    assert vec.size == len(FEATURE_NAMES)
    return vec


def as_dict(vec) -> dict:
    return {name: round(float(v), 4) for name, v in zip(FEATURE_NAMES, vec)}
