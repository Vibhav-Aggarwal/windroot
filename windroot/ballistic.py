"""Ballistic back-mapping of in-situ solar wind to the PFSS source surface.

The ballistic approximation assumes a solar-wind parcel travels radially outward
at constant speed from the source surface (R_ss). During the travel time the Sun
rotates, so in the corotating Carrington frame the parcel's source-surface footpoint
sits at a *higher* Carrington longitude than the spacecraft connection point:

    Delta_lon = Omega_sun * (r_obs - r_ss) / v_sw

Latitude is assumed conserved (purely radial flow). The dominant systematic
uncertainty is the choice of source-surface height R_ss (Koukras et al. 2022;
Dakeyo et al. 2024), which we expose and Monte-Carlo over.

References
----------
- Nolte & Roelof (1973) — ballistic approximation.
- Macneil et al. (2022), Koukras et al. (2022), Dakeyo et al. (2024) — accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._units import (
    AU_IN_RSUN,
    DEFAULT_R_SS,
    RSUN_KM,
    SIDEREAL_ROTATION_PERIOD,
    as_kms,
    as_rsun,
    sidereal_rate_deg_per_s,
    wrap_longitude,
)


def corotation_shift(v_sw, r_obs, r_ss=DEFAULT_R_SS, period=SIDEREAL_ROTATION_PERIOD) -> float:
    """Longitudinal corotation shift (degrees) from r_ss out to r_obs.

    Parameters
    ----------
    v_sw : float or Quantity
        Radial solar-wind speed (km/s if plain float).
    r_obs : float or Quantity
        Observation heliocentric distance (Rsun if plain float).
    r_ss : float
        Source-surface radius in Rsun.
    period : Quantity
        Solar sidereal rotation period.
    """
    v = as_kms(v_sw)
    r = as_rsun(r_obs)
    dist_km = (r - r_ss) * RSUN_KM
    travel_time_s = dist_km / v
    return sidereal_rate_deg_per_s(period) * travel_time_s


@dataclass
class BallisticResult:
    """A back-mapped source-surface footpoint."""

    lon_ss: float          # Carrington longitude at source surface [deg]
    lat_ss: float          # latitude at source surface [deg] (== lat_obs)
    r_ss: float            # source-surface radius [Rsun]
    delta_lon: float       # applied corotation shift [deg]
    travel_time_days: float
    v_sw: float            # km/s


def ballistic_backmap(
    v_sw,
    r_obs,
    lon_obs: float,
    lat_obs: float,
    r_ss: float = DEFAULT_R_SS,
    period=SIDEREAL_ROTATION_PERIOD,
) -> BallisticResult:
    """Map a single in-situ measurement back to the source surface.

    lon_obs/lat_obs are Carrington coordinates of the spacecraft (degrees).
    """
    v = as_kms(v_sw)
    r = as_rsun(r_obs)
    dlon = corotation_shift(v, r, r_ss=r_ss, period=period)
    dist_km = (r - r_ss) * RSUN_KM
    travel_days = dist_km / v / 86400.0
    return BallisticResult(
        lon_ss=wrap_longitude(lon_obs + dlon),
        lat_ss=float(lat_obs),
        r_ss=float(r_ss),
        delta_lon=float(dlon),
        travel_time_days=float(travel_days),
        v_sw=float(v),
    )


@dataclass
class BallisticEnsemble:
    """Monte-Carlo ensemble of back-mapped source-surface footpoints."""

    lon_ss: np.ndarray     # [deg], wrapped, shape (n,)
    lat_ss: np.ndarray     # [deg]
    r_ss: np.ndarray       # [Rsun]
    v_sw: np.ndarray       # [km/s]

    def summary(self) -> dict:
        """Circular-mean longitude, mean latitude, and spreads."""
        lon_rad = np.deg2rad(self.lon_ss)
        mean_lon = np.rad2deg(np.arctan2(np.sin(lon_rad).mean(), np.cos(lon_rad).mean())) % 360.0
        # circular std (Mardia)
        R = np.hypot(np.sin(lon_rad).mean(), np.cos(lon_rad).mean())
        lon_std = np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-12))))
        return {
            "lon_ss_mean": float(mean_lon),
            "lon_ss_std": float(lon_std),
            "lat_ss_mean": float(self.lat_ss.mean()),
            "lat_ss_std": float(self.lat_ss.std()),
            "r_ss_mean": float(self.r_ss.mean()),
        }


def ballistic_backmap_mc(
    v_sw,
    r_obs,
    lon_obs: float,
    lat_obs: float,
    v_sw_err=0.0,
    lat_err: float = 0.0,
    r_ss_range=(1.5, 3.5),
    n: int = 2000,
    period=SIDEREAL_ROTATION_PERIOD,
    rng: Optional[np.random.Generator] = None,
) -> BallisticEnsemble:
    """Monte-Carlo back-mapping propagating speed and source-surface-height error.

    The source-surface height is the dominant systematic, so it is sampled
    uniformly across ``r_ss_range`` rather than fixed.
    """
    rng = rng or np.random.default_rng()
    v0 = as_kms(v_sw)
    dv = as_kms(v_sw_err) if v_sw_err else 0.0
    r = as_rsun(r_obs)

    v_samples = (v0 + dv * rng.standard_normal(n)) if dv > 0 else np.full(n, v0)
    v_samples = np.clip(v_samples, 50.0, 3000.0)  # physical solar-wind bounds
    r_ss_samples = rng.uniform(r_ss_range[0], r_ss_range[1], n)
    lat_samples = np.full(n, float(lat_obs))
    if lat_err:
        lat_samples = lat_samples + lat_err * rng.standard_normal(n)

    dist_km = (r - r_ss_samples) * RSUN_KM
    travel_s = dist_km / v_samples
    dlon = sidereal_rate_deg_per_s(period) * travel_s
    lon_ss = (lon_obs + dlon) % 360.0

    return BallisticEnsemble(
        lon_ss=lon_ss,
        lat_ss=lat_samples,
        r_ss=r_ss_samples,
        v_sw=v_samples,
    )


def au_to_rsun(r_au: float) -> float:
    """Convenience: AU -> Rsun."""
    return r_au * AU_IN_RSUN
