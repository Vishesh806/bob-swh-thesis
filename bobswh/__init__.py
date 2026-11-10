"""Multi-step significant wave height forecasting in the Bay of Bengal.

Phase 1 (ingestion, preprocessing, spatial alignment) and Phase 2 (feature
engineering, sliding windows, chronological splits), with the models,
evaluation and live forecasting in their own modules.

The public names below are imported lazily, on first use, so that light
modules (features, trees, live) can be used without the heavy ingestion
dependencies (xarray, scipy), e.g. by the dashboard.
"""
from importlib import import_module

_EXPORTS = {
    "NearestNodeAligner": "alignment", "haversine_km": "alignment",
    "BuoyFileSpec": "buoy", "BuoyReader": "buoy",
    "ERA5Downloader": "era5", "ERA5PointExtractor": "era5",
    "FeatureEngineer": "features",
    "Phase12Pipeline": "pipeline",
    "Preprocessor": "preprocess",
    "ChronologicalSplitter": "windowing", "SlidingWindowGenerator": "windowing", "WindowSpec": "windowing",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
