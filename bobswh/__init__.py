"""Multi-step significant wave height forecasting in the Bay of Bengal.

Phase 1 (ingestion, preprocessing, spatial alignment) and Phase 2 (feature
engineering, sliding windows, chronological splits).
"""
from .alignment import NearestNodeAligner, haversine_km
from .buoy import BuoyFileSpec, BuoyReader
from .era5 import ERA5Downloader, ERA5PointExtractor
from .features import FeatureEngineer
from .pipeline import Phase12Pipeline
from .preprocess import Preprocessor
from .windowing import ChronologicalSplitter, SlidingWindowGenerator, WindowSpec

__all__ = [
    "BuoyFileSpec",
    "BuoyReader",
    "ChronologicalSplitter",
    "ERA5Downloader",
    "ERA5PointExtractor",
    "FeatureEngineer",
    "NearestNodeAligner",
    "Phase12Pipeline",
    "Preprocessor",
    "SlidingWindowGenerator",
    "WindowSpec",
    "haversine_km",
]
