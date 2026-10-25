"""Spatial alignment of point observations (buoys) with the ERA5 grid.

The grid nodes are embedded on the unit sphere and indexed with a KD-tree.
Euclidean (chord) distance between two points on the unit sphere is
c = 2 sin(d / 2R), a strictly increasing function of the great-circle distance
d, so the KD-tree nearest neighbour is exactly the haversine nearest neighbour.
A naive Euclidean metric on (lat, lon) would instead overweight longitude by
1 / cos(lat), about 3.5 % at 15 N, and can pick the wrong node.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0088   # IUGG mean Earth radius


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between points given in decimal degrees."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def lonlat_to_xyz(lat, lon) -> np.ndarray:
    """Unit-sphere Cartesian coordinates, shape (n, 3)."""
    phi, lmb = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.column_stack([np.cos(phi) * np.cos(lmb), np.cos(phi) * np.sin(lmb), np.sin(phi)])


def grid_key(lat: np.ndarray, lon: np.ndarray, valid: np.ndarray) -> str:
    """Hash that identifies a grid together with its valid-node mask."""
    h = hashlib.sha1()
    for arr in (np.asarray(lat, float), np.asarray(lon, float), np.asarray(valid, bool)):
        h.update(np.ascontiguousarray(arr).tobytes())
        h.update(str(arr.shape).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class NodeMatch:
    """Grid nodes matched to one site and their interpolation weights."""

    site: str
    iy: np.ndarray            # latitude indices, shape (k,)
    ix: np.ndarray            # longitude indices, shape (k,)
    lat: np.ndarray
    lon: np.ndarray
    distance_km: np.ndarray
    weights: np.ndarray       # sum to 1


class NearestNodeAligner:
    """Map arbitrary points to the nearest *valid* nodes of a regular lat/lon grid.

    Parameters
    ----------
    lat, lon:
        1-D coordinate vectors of the grid (degrees).
    valid:
        Boolean mask of shape (len(lat), len(lon)). Nodes marked False (land,
        or NaN in the wave model) are never returned. This matters for ERA5
        wave fields, which are undefined over land: the geometrically nearest
        node to a coastal buoy can be a land node.
    """

    def __init__(self, lat: np.ndarray, lon: np.ndarray, valid: np.ndarray | None = None) -> None:
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        if valid is None:
            valid = np.ones(lat2d.shape, bool)
        valid = np.asarray(valid, bool)
        if valid.shape != lat2d.shape:
            raise ValueError(f"valid mask shape {valid.shape} does not match grid {lat2d.shape}")
        self._iy, self._ix = np.nonzero(valid)
        if self._iy.size == 0:
            raise ValueError("the grid has no valid nodes")
        self._lat = lat2d[self._iy, self._ix]
        self._lon = lon2d[self._iy, self._ix]
        self._tree = cKDTree(lonlat_to_xyz(self._lat, self._lon))

    @property
    def n_valid(self) -> int:
        return int(self._iy.size)

    def query(
        self,
        site: str,
        lat: float,
        lon: float,
        k: int = 1,
        power: float = 2.0,
        max_distance_km: float | None = None,
    ) -> NodeMatch:
        """Return the k nearest valid nodes and inverse-distance weights.

        With k = 1 the single nearest node gets weight 1. With k > 1 the weights
        are w_i = d_i^-p / sum_j d_j^-p (inverse-distance weighting); a node
        closer than 1 m takes all the weight.

        Raises
        ------
        ValueError
            If the nearest valid node is farther than ``max_distance_km``.
        """
        k = int(min(max(k, 1), self.n_valid))
        _, idx = self._tree.query(lonlat_to_xyz([lat], [lon])[0], k=k)
        idx = np.atleast_1d(idx)
        dist = haversine_km(lat, lon, self._lat[idx], self._lon[idx])
        order = np.argsort(dist)
        idx, dist = idx[order], dist[order]
        if max_distance_km is not None and dist[0] > max_distance_km:
            raise ValueError(
                f"{site}: nearest valid grid node is {dist[0]:.1f} km away "
                f"(limit {max_distance_km} km); check the buoy position and the land mask"
            )
        if k == 1 or dist[0] < 1e-3:
            weights = np.zeros(k)
            weights[0] = 1.0
        else:
            inv = dist ** -power
            weights = inv / inv.sum()
        return NodeMatch(
            site=site,
            iy=self._iy[idx],
            ix=self._ix[idx],
            lat=self._lat[idx],
            lon=self._lon[idx],
            distance_km=dist,
            weights=weights,
        )
