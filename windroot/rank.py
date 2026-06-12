"""Fuse magnetic connectivity with spectroscopic outflow to rank source regions.

Given a Monte-Carlo cloud of photospheric footpoints (from ballistic back-mapping
+ PFSS), candidate source regions are found by binning the cloud on a
longitude/latitude grid. Each candidate is scored by:

- **connectivity** : the fraction of MC footpoints in/near the candidate cell
  (how strongly the magnetic mapping points there, given its uncertainty);
- **outflow**      : the observed spectroscopic upflow speed at the candidate
  (does plasma actually flow outward there? — the novel evidence layer).

The combined confidence is a weighted, normalised blend. With no Doppler map the
ranking reduces to the classical connectivity-only result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .spectro import DopplerMap


@dataclass
class SourceCandidate:
    lon: float                 # candidate centre longitude [deg]
    lat: float                 # candidate centre latitude [deg]
    connectivity_score: float  # in [0, 1], relative to the densest candidate in this call
    outflow_score: float       # 0.5 if no Doppler map was provided (neutral); in [0, 1]
                               # normalised across covered candidates when a map IS
                               # provided; NaN if the candidate lies outside the map's
                               # spectroscopic coverage (no evidence either way)
    outflow_kms: float         # raw sampled upflow [km/s] (NaN if none)
    confidence: float          # combined, in [0, 1]
    n_samples: int             # MC footpoints attributed to this candidate

    @property
    def has_outflow_evidence(self) -> bool:
        """True iff a Doppler map was supplied AND covered this candidate."""
        import math
        return not math.isnan(self.outflow_score) and self.outflow_score != 0.5


def rank_sources(
    lon_phot: np.ndarray,
    lat_phot: np.ndarray,
    doppler_map: Optional[DopplerMap] = None,
    bin_deg: float = 5.0,
    sample_radius_deg: float = 5.0,
    w_connectivity: float = 0.5,
    w_outflow: float = 0.5,
    min_fraction: float = 0.02,
) -> list[SourceCandidate]:
    """Rank candidate solar-wind source regions.

    Parameters
    ----------
    lon_phot, lat_phot : np.ndarray
        Photospheric footpoint Monte-Carlo cloud (degrees).
    doppler_map : DopplerMap, optional
        Spectroscopic outflow evidence. If ``None``, ranking is connectivity-only.
    bin_deg : float
        Grid cell size for clustering the footpoint cloud.
    sample_radius_deg : float
        Neighbourhood radius for sampling outflow at each candidate.
    w_connectivity, w_outflow : float
        Relative weights (renormalised internally).
    min_fraction : float
        Drop candidates holding less than this fraction of the cloud.
    """
    lon_phot = np.asarray(lon_phot, dtype=float)
    lat_phot = np.asarray(lat_phot, dtype=float)
    # closed-line / failed traces come back as NaN; drop them before binning
    good = np.isfinite(lon_phot) & np.isfinite(lat_phot)
    lon_phot = lon_phot[good]
    lat_phot = lat_phot[good]
    n_total = lon_phot.size
    if n_total == 0:
        return []

    # bin onto a regular grid
    lon_idx = np.floor(lon_phot / bin_deg).astype(int)
    lat_idx = np.floor((lat_phot + 90.0) / bin_deg).astype(int)
    keys = lon_idx * 100000 + lat_idx
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)

    if w_connectivity + w_outflow <= 0:
        w_connectivity, w_outflow = 0.5, 0.5
    wsum = w_connectivity + w_outflow
    wc, wo = w_connectivity / wsum, w_outflow / wsum

    candidates: list[SourceCandidate] = []
    max_count = counts.max()
    for k, cell_key in enumerate(uniq):
        sel = inverse == k
        n = int(sel.sum())
        if n / n_total < min_fraction:
            continue
        # circular-mean longitude, mean latitude of footpoints in this cell
        lon_rad = np.deg2rad(lon_phot[sel])
        c_lon = np.rad2deg(np.arctan2(np.sin(lon_rad).mean(), np.cos(lon_rad).mean())) % 360.0
        c_lat = float(lat_phot[sel].mean())
        conn = n / max_count  # normalised to the densest cell

        if doppler_map is not None:
            up = doppler_map.sample(c_lon, c_lat, radius_deg=sample_radius_deg)
        else:
            up = float("nan")

        candidates.append(
            SourceCandidate(
                lon=float(c_lon),
                lat=c_lat,
                connectivity_score=float(conn),
                outflow_score=0.5,        # filled below once we know the outflow range
                outflow_kms=float(up),
                confidence=0.0,           # filled below
                n_samples=n,
            )
        )

    if not candidates:
        return []

    # normalise outflow across candidates (only where measured)
    ups = np.array([c.outflow_kms for c in candidates], dtype=float)
    measured = np.isfinite(ups)
    if doppler_map is not None and measured.any():
        pos = np.clip(ups, 0.0, None)  # only outflow (positive) counts as evidence
        lo, hi = np.nanmin(pos[measured]), np.nanmax(pos[measured])
        rng = hi - lo if hi > lo else 1.0
        for c in candidates:
            if np.isfinite(c.outflow_kms):
                c.outflow_score = float(np.clip((max(c.outflow_kms, 0.0) - lo) / rng, 0.0, 1.0))
            else:
                # candidate lies outside the Doppler map's coverage -> no evidence
                # either way; do NOT collapse to 0 (which would penalise it as if it
                # were a confirmed downflow). NaN here is read by the confidence loop
                # below to fall back to connectivity-only for this candidate.
                c.outflow_score = float("nan")
        for c in candidates:
            if np.isnan(c.outflow_score):
                c.confidence = c.connectivity_score
            else:
                c.confidence = wc * c.connectivity_score + wo * c.outflow_score
    else:
        # connectivity only
        for c in candidates:
            c.outflow_score = 0.5
            c.confidence = c.connectivity_score

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates
