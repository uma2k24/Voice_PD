"""Shared train/serve acoustic measurements on unnormalised mono PCM.

Jitter/shimmer are local percentages (Praat fractions multiplied by 100).
CPPS uses Praat PowerCepstrogram smoothing, not the legacy CPP approximation.
GNE is the maximum of Praat's band-envelope correlation matrix, a ratio.
"""
from __future__ import annotations

import math
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
FEATURE_NAMES = [
    "jitter_local_percent", "shimmer_local_percent", "hnr_db", "cpps_db",
    "gne_ratio", "f0_mean_hz", "f0_std_hz", "f1_mean_hz", "f2_mean_hz",
    "f3_mean_hz", "mean_period_seconds",
]
UNITS = dict(zip(FEATURE_NAMES, ["%", "%", "dB", "dB", "ratio", "Hz", "Hz",
                               "Hz", "Hz", "Hz", "s"]))
EXTRACTOR_VERSION = "praat-conversation-v1"
# These settings are shared by training and live inference.
SETTINGS = {"sample_rate": SAMPLE_RATE, "pitch_floor_hz": 60,
            "pitch_ceiling_hz": 500, "cpps_time_smoothing_s": .01,
            "cpps_quefrency_smoothing_s": .001,
            "gne_min_hz": 500, "gne_max_hz": 4500,
            "gne_bandwidth_hz": 1000, "gne_step_hz": 80}


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def extract_acoustics(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    y = np.asarray(samples, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError("Expected finite, nonempty mono audio.")
    if sample_rate < 10000 or sample_rate > 96000:
        raise ValueError("Sample rate must be 10000-96000 Hz.")
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        y = resample_poly(y, SAMPLE_RATE // divisor, sample_rate // divisor)
    duration = y.size / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(y ** 2)))
    clipped = float(np.mean(np.abs(y) >= .999))
    features = dict.fromkeys(FEATURE_NAMES)
    quality = {"duration_seconds": duration, "rms": rms,
               "clipped_fraction": clipped, "voiced_fraction": 0.0,
               "usable": False, "reasons": []}
    result = {"features": features, "units": UNITS, "quality": quality,
              "extractor_version": EXTRACTOR_VERSION}
    if duration < 1:
        quality["reasons"].append("too_short")
        return result
    if rms < .003:
        quality["reasons"].append("too_quiet")
        return result
    if clipped > .01:
        quality["reasons"].append("clipping")
    sound = parselmouth.Sound(y - y.mean(), sampling_frequency=SAMPLE_RATE)
    pitch = sound.to_pitch_ac(time_step=.01, pitch_floor=60, pitch_ceiling=500)
    frequencies = pitch.selected_array["frequency"]
    voiced = frequencies[frequencies > 0]
    quality["voiced_fraction"] = float(voiced.size / max(1, frequencies.size))
    if voiced.size < 10 or quality["voiced_fraction"] < .3:
        quality["reasons"].append("insufficient_voicing")
        return result
    features["f0_mean_hz"] = finite(voiced.mean())
    features["f0_std_hz"] = finite(voiced.std())

    def measure(name, function):
        try:
            features[name] = finite(function())
        except parselmouth.PraatError:
            features[name] = None

    pulses = call([sound, pitch], "To PointProcess (cc)")
    periods = (0, 0, 1 / 500, 1 / 60, 1.3)
    measure("jitter_local_percent", lambda: 100 * call(pulses, "Get jitter (local)", *periods))
    measure("shimmer_local_percent", lambda: 100 * call(
        [sound, pulses], "Get shimmer (local)", *periods, 1.6))
    measure("mean_period_seconds", lambda: call(pulses, "Get mean period", *periods))
    measure("hnr_db", lambda: call(sound.to_harmonicity_cc(
        time_step=.01, minimum_pitch=60, silence_threshold=.1,
        periods_per_window=1), "Get mean", 0, 0))
    measure("cpps_db", lambda: call(
        call(sound, "To PowerCepstrogram", 60, .002, 5000, 50),
        "Get CPPS", False, .01, .001, 60, 500, .05, "Parabolic", .001, 0,
        "Straight", "Robust"))
    measure("gne_ratio", lambda: np.max(call(
        sound, "To Harmonicity (gne)", 500, 4500, 1000, 80).values))
    formants = sound.to_formant_burg(time_step=.01, max_number_of_formants=5,
                                    maximum_formant=5000, window_length=.025)
    for number in (1, 2, 3):
        values = [formants.get_value_at_time(number, float(t)) for t in pitch.xs()
                  if pitch.get_value_at_time(float(t)) > 0]
        valid = np.asarray(values)[np.isfinite(values)]
        features[f"f{number}_mean_hz"] = finite(valid.mean()) if valid.size else None
    if any(features[name] is None for name in FEATURE_NAMES[:5]):
        quality["reasons"].append("missing_core_measurement")
    quality["usable"] = not quality["reasons"]
    return result


class RollingAudio:
    """Bounded 4-second window, updated each second after initial fill."""
    def __init__(self, window_seconds=4, hop_seconds=1):
        self.capacity = int(window_seconds * SAMPLE_RATE)
        self.hop = int(hop_seconds * SAMPLE_RATE)
        self.samples = np.empty(0, dtype=np.float64)
        self.total = 0
        self.last_emitted = 0

    def append(self, pcm: bytes):
        if not pcm or len(pcm) % 2 or len(pcm) > SAMPLE_RATE * 2:
            raise ValueError("Audio must be 16-bit mono PCM, at most one second per message.")
        y = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
        self.samples = np.concatenate((self.samples, y))[-self.capacity:]
        self.total += y.size
        if self.samples.size < self.capacity or self.total - self.last_emitted < self.hop:
            return None
        self.last_emitted = self.total
        return self.samples.copy(), self.total / SAMPLE_RATE

    def clear(self):
        # Preserve timeline, discard a mixed/agent-playback window.
        self.samples = np.empty(0, dtype=np.float64)
        self.last_emitted = self.total
