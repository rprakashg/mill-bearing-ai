"""Constants shared by training, the pipeline, and the edge scorer.

Keeping these in one place is the point of the demo: the edge computes features
with exactly the same code and order that the model was trained on.
"""

SAMPLE_RATE_HZ = 5120          # vibration sensor sample rate
WINDOW_SECONDS = 1.0           # one feature vector per second of signal
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)

CLASSES = ["healthy", "imbalance", "bearing_wear"]

FEATURE_NAMES = [
    "rms_g",            # overall vibration energy
    "peak_g",           # largest absolute sample
    "crest_factor",     # peak / rms, rises with impacts
    "kurtosis",         # "spikiness", classic bearing-defect indicator
    "band_0_100_g",     # around 1x running speed (imbalance)
    "band_100_500_g",   # harmonics, looseness
    "band_500_1500_g",  # mid band
    "band_1500_2560_g", # bearing resonance band (defect impacts ring here)
    "temp_c",           # bearing housing temperature
    "current_a",        # motor current
]
N_FEATURES = len(FEATURE_NAMES)

MODEL_NAME = "bearing"
ONNX_INPUT = "features"
ONNX_OUTPUT = "probabilities"
