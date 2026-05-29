"""Spectroscopic outflow evidence — the novel input layer.

A :class:`DopplerMap` holds plasma outflow (upflow) speed on a Carrington
longitude/latitude grid, derived from EUV spectroscopy (SPICE / Hinode EIS).
Sign convention: ``upflow`` is **positive outward** (a blueshift on the disk),
because outflowing plasma is the candidate seed of the solar wind.

Loaders for real SPICE/EIS fitted cubes are provided as optional, lazily-imported
helpers; the core only needs gridded arrays. If a clean SPICE Doppler map is
wanted, Arpit's A&A 706 skew-correction can be applied upstream (see
:mod:`windroot.spectro.apply_skew_correction`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DopplerMap:
    """Gridded spectroscopic outflow map in Carrington coordinates.

    Parameters
    ----------
    lon, lat : np.ndarray
        1-D monotonic grid axes in degrees. ``lon`` in [0, 360), ``lat`` in [-90, 90].
    upflow : np.ndarray
        2-D array, shape (lat.size, lon.size), outflow speed in km/s (positive = up).
    line : str
        Spectral line label (e.g. "Ne VIII 770").
    """

    lon: np.ndarray
    lat: np.ndarray
    upflow: np.ndarray
    line: str = ""

    def __post_init__(self):
        self.lon = np.asarray(self.lon, dtype=float)
        self.lat = np.asarray(self.lat, dtype=float)
        self.upflow = np.asarray(self.upflow, dtype=float)
        if self.upflow.shape != (self.lat.size, self.lon.size):
            raise ValueError(
                f"upflow shape {self.upflow.shape} != (lat={self.lat.size}, lon={self.lon.size})"
            )

    @classmethod
    def from_vlos(cls, lon, lat, vlos, line: str = "", blueshift_is_outflow: bool = True):
        """Build from a line-of-sight velocity map.

        For near-disk-centre observations, a blueshift (negative v_los) corresponds
        to outflow; set ``blueshift_is_outflow=False`` to keep the raw sign.
        """
        vlos = np.asarray(vlos, dtype=float)
        upflow = -vlos if blueshift_is_outflow else vlos
        return cls(lon=lon, lat=lat, upflow=upflow, line=line)

    def sample(self, lon: float, lat: float, radius_deg: float = 5.0) -> float:
        """Mean outflow within ``radius_deg`` of a point (great-circle distance).

        Returns ``np.nan`` if no grid points fall in the neighbourhood.
        """
        d = _angular_distance(lon, lat, self._lon2d(), self._lat2d())
        mask = np.isfinite(self.upflow) & (d <= radius_deg)
        if not mask.any():
            return float("nan")
        return float(np.nanmean(self.upflow[mask]))

    def sample_many(self, lons, lats, radius_deg: float = 5.0) -> np.ndarray:
        return np.array([self.sample(lo, la, radius_deg) for lo, la in zip(lons, lats)])

    def _lon2d(self):
        return np.broadcast_to(self.lon[None, :], self.upflow.shape)

    def _lat2d(self):
        return np.broadcast_to(self.lat[:, None], self.upflow.shape)


def _angular_distance(lon0, lat0, lon, lat) -> np.ndarray:
    """Great-circle angular distance (deg) between (lon0,lat0) and arrays (lon,lat)."""
    lo0, la0 = np.deg2rad(lon0), np.deg2rad(lat0)
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    dlon = lo - lo0
    cos_d = np.sin(la0) * np.sin(la) + np.cos(la0) * np.cos(la) * np.cos(dlon)
    return np.rad2deg(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def apply_skew_correction(spice_dat, spice_hdr, xlshift, ylshift, **kwargs):
    """Optional pre-clean of a SPICE L2 cube using Arpit's A&A 706 skew correction.

    Lazily imports the user's ``spice-line-fits`` modules if importable; this honours
    and builds on his published Doppler-artifact correction as an upstream step.
    Returns the correction result dict (see spice-line-fits.full_correction).
    """
    try:
        from skew_correction import full_correction  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires spice-line-fits on path
        raise ImportError(
            "apply_skew_correction needs Shrivastav's spice-line-fits modules on PYTHONPATH "
            "(github.com/ArpitkShrivastav/spice-line-fits)."
        ) from exc
    return full_correction(spice_dat, spice_hdr, xlshift, ylshift, **kwargs)
