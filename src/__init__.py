"""Parkinson's Voice Analyser - v2."""
def __getattr__(name):
    # The conversation backend does not need the legacy librosa/nonlinear stack.
    if name in {"extract_features", "FEATURE_NAMES"}:
        from . import feature_extractor
        return getattr(feature_extractor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
