"""Adapter that lets ``windroot.find_sources`` use a sunkit-magex PFSS solution.

Builds a thin wrapper exposing ``map_footpoints(lon_ss, lat_ss) -> (lon, lat)``
(the field-mapper interface windroot expects), so an arbitrary sunpy synoptic
magnetogram can drive the connectivity path.

Lazy imports: this module requires the ``pfss`` extra (sunpy + sunkit-magex +
streamtracer). Failures point at the missing extra.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PFSSAdapter:
    """Trace seeds at the source surface down to the photosphere with sunkit-magex."""

    pfss_out: object         # sunkit_magex.pfss.Output
    r_ss: float = 2.5
    tracer: Optional[object] = None    # default: FortranTracer if available

    def map_footpoints(self, lon_ss, lat_ss):
        from astropy import constants as const
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from sunkit_magex.pfss.tracing import PythonTracer

        lon = np.atleast_1d(np.asarray(lon_ss, dtype=float))
        lat = np.atleast_1d(np.asarray(lat_ss, dtype=float))

        # Seed *just inside* the source surface; exactly at R_ss the seed sits on
        # the outer domain boundary and the tracer cannot start.
        r_seed = self.r_ss * 0.999
        seeds = SkyCoord(
            lon * u.deg,
            lat * u.deg,
            r_seed * const.R_sun,
            frame=self.pfss_out.coordinate_frame,
        )
        tracer = self.tracer or PythonTracer(atol=1e-5, rtol=1e-5)
        flines = tracer.trace(seeds, self.pfss_out)

        lon_phot = np.full(lon.size, np.nan)
        lat_phot = np.full(lon.size, np.nan)
        for i, fl in enumerate(flines):
            if not getattr(fl, "is_open", True):
                continue
            fp = fl.solar_footpoint
            if fp is None:
                continue
            try:
                ifp = fp.transform_to(self.pfss_out.coordinate_frame)
            except Exception:
                ifp = fp
            lon_phot[i] = float(ifp.lon.to_value(u.deg)) % 360.0
            lat_phot[i] = float(ifp.lat.to_value(u.deg))
        return lon_phot, lat_phot


def synthetic_dipole_synoptic(b_polar: float = 10.0, nlon: int = 180, nlat: int = 90):
    """Build a synthetic dipolar B_r synoptic magnetogram as a sunpy.Map.

    B_r(theta) = b_polar * cos(theta), aligned with the rotation axis. Used by
    Backtest B to cross-validate the windroot analytic dipole against
    sunkit-magex's numerical PFSS on the same field.
    """
    import sunpy.map
    from sunkit_magex.pfss.utils import carr_cea_wcs_header
    from astropy.time import Time

    sin_lat_edges = np.linspace(-1.0, 1.0, nlat + 1)
    sin_lat_centers = 0.5 * (sin_lat_edges[:-1] + sin_lat_edges[1:])
    br = (b_polar * sin_lat_centers[:, None]) * np.ones((nlat, nlon))

    header = carr_cea_wcs_header(Time("2020-01-01T00:00:00"), (nlon, nlat))
    return sunpy.map.Map(br, header)
