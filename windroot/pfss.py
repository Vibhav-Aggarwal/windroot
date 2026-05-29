"""Map source-surface footpoints down to the photosphere via PFSS.

Two backends:

1. :class:`DipoleSourceSurface` — an *analytic* axisymmetric dipole + source-surface
   potential field. It gives a closed-form footpoint mapping (used as ground truth
   in tests) and a magnetic field for the generic numeric tracer. No external deps.

2. :func:`pfss_connectivity` — a thin wrapper around ``sunkit-magex`` for real
   magnetograms (GONG/ADAPT). Imported lazily; optional.

Analytic dipole (l=1, source-surface boundary B_theta(R_ss)=0):

    B_r    = -a (1 + 2 R_ss^3 / r^3) cos(theta)
    B_theta=  a (1 -   R_ss^3 / r^3) sin(theta)

Field-line invariant (theta = colatitude, phi conserved):

    (r^2 + 2 R_ss^3 / r) sin^2(theta) = const

so a source-surface point at colatitude theta_ss maps to the photosphere at

    sin^2(theta_phot) = 3 R_ss^2 sin^2(theta_ss) / (R_sun^2 + 2 R_ss^3 / R_sun)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from ._units import DEFAULT_R_SS


@dataclass
class Footpoint:
    """A photospheric footpoint with its source-surface origin."""

    lon: float       # Carrington longitude [deg]
    lat: float       # latitude [deg]
    open_field: bool = True


class DipoleSourceSurface:
    """Analytic axisymmetric dipole potential field with a source surface.

    Parameters
    ----------
    r_ss : float
        Source-surface radius in Rsun.
    b_polar : float
        Photospheric polar field strength (sets the field amplitude/sign; the
        connectivity mapping is independent of it, but it is used by the tracer).
    r_sun : float
        Photospheric radius in Rsun (1.0 by definition; exposed for testing).
    """

    def __init__(self, r_ss: float = DEFAULT_R_SS, b_polar: float = 1.0, r_sun: float = 1.0):
        self.r_ss = float(r_ss)
        self.r_sun = float(r_sun)
        # B_r(r_sun, theta=0) = -a (1 + 2 r_ss^3/r_sun^3) = b_polar  =>  a = -b_polar/(...)
        self._a = -b_polar / (1.0 + 2.0 * self.r_ss**3 / self.r_sun**3)

    # --- magnetic field (spherical, axisymmetric) ---
    def b_r(self, r, theta):
        return -self._a * (1.0 + 2.0 * self.r_ss**3 / r**3) * np.cos(theta)

    def b_theta(self, r, theta):
        return self._a * (1.0 - self.r_ss**3 / r**3) * np.sin(theta)

    # --- closed-form footpoint mapping (the ground truth) ---
    def footpoint(self, lon_ss: float, lat_ss: float) -> Footpoint:
        """Trace a source-surface point down to the photosphere analytically."""
        theta_ss = np.deg2rad(90.0 - lat_ss)
        denom = self.r_sun**2 + 2.0 * self.r_ss**3 / self.r_sun
        sin2 = 3.0 * self.r_ss**2 * np.sin(theta_ss) ** 2 / denom
        sin2 = float(np.clip(sin2, 0.0, 1.0))
        theta_phot = np.arcsin(np.sqrt(sin2))
        # preserve hemisphere
        if lat_ss < 0:
            lat_phot = -(90.0 - np.rad2deg(theta_phot))
        else:
            lat_phot = 90.0 - np.rad2deg(theta_phot)
        return Footpoint(lon=float(lon_ss) % 360.0, lat=float(lat_phot), open_field=True)

    def trace_footpoint(self, lon_ss: float, lat_ss: float) -> Footpoint:
        """Trace down numerically by integrating dtheta/dr from R_ss to R_sun.

        Independent cross-check of the closed-form :meth:`footpoint`. Integrating in
        radius (rather than colatitude) is well-behaved because B_theta -> 0 at the
        source surface, so dtheta/dr = 0 there (no singularity).
        """
        theta0 = np.deg2rad(90.0 - lat_ss)

        def dtheta_dr(r, theta):
            num = 1.0 - self.r_ss**3 / r**3
            den = r * (1.0 + 2.0 * self.r_ss**3 / r**3)
            return [-num * np.tan(theta[0]) / den]

        sol = solve_ivp(
            dtheta_dr,
            (self.r_ss, self.r_sun),
            [theta0],
            max_step=0.01,
            rtol=1e-9,
            atol=1e-11,
        )
        theta_phot = float(sol.y[0, -1])
        lat_phot = 90.0 - np.rad2deg(theta_phot)
        return Footpoint(lon=float(lon_ss) % 360.0, lat=float(lat_phot), open_field=True)


def map_footpoints(field: DipoleSourceSurface, lon_ss, lat_ss) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised analytic mapping of arrays of source-surface points -> photosphere.

    Returns (lon_phot, lat_phot) arrays in degrees.
    """
    lon_ss = np.atleast_1d(np.asarray(lon_ss, dtype=float))
    lat_ss = np.atleast_1d(np.asarray(lat_ss, dtype=float))
    theta_ss = np.deg2rad(90.0 - lat_ss)
    denom = field.r_sun**2 + 2.0 * field.r_ss**3 / field.r_sun
    sin2 = np.clip(3.0 * field.r_ss**2 * np.sin(theta_ss) ** 2 / denom, 0.0, 1.0)
    theta_phot = np.arcsin(np.sqrt(sin2))
    lat_phot = np.where(lat_ss < 0, -(90.0 - np.rad2deg(theta_phot)), 90.0 - np.rad2deg(theta_phot))
    return lon_ss % 360.0, lat_phot


def pfss_connectivity(gong_map, r_ss: float = DEFAULT_R_SS, nrho: int = 35):
    """Build a sunkit-magex PFSS extrapolation from a magnetogram (optional backend).

    Lazily imports ``sunkit_magex``. Returns the PFSS output object; tracing of
    individual field lines is delegated to sunkit-magex tracers by the caller.
    Raises a clear error if the optional dependency is missing.
    """
    try:
        from sunkit_magex import pfss  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only with extra installed
        raise ImportError(
            "pfss_connectivity requires the 'pfss' extra: pip install 'windroot[pfss]'"
        ) from exc
    pfss_in = pfss.Input(gong_map, nrho, r_ss)
    return pfss.pfss(pfss_in)
