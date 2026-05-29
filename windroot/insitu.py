"""In-situ solar-wind measurements to back-map.

The core only needs a small record: where the spacecraft was (Carrington coords +
heliocentric distance) and the measured radial wind speed. Real loaders (OMNI via
CDAWeb, PSP/Wind) are optional and lazily imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._units import AU_IN_RSUN, as_kms, as_rsun


@dataclass
class InSituStream:
    """A single in-situ solar-wind sample to back-map.

    Parameters
    ----------
    v_sw : float
        Radial solar-wind speed [km/s].
    r_obs : float
        Heliocentric distance [Rsun]. Use :func:`from_au` for AU input.
    lon_obs, lat_obs : float
        Spacecraft Carrington longitude/latitude [deg].
    v_sw_err : float
        1-sigma speed uncertainty [km/s].
    label : str
        Free-form description (spacecraft, time, etc.).
    """

    v_sw: float
    r_obs: float
    lon_obs: float
    lat_obs: float
    v_sw_err: float = 0.0
    label: str = ""

    @classmethod
    def from_au(cls, v_sw, r_au, lon_obs, lat_obs, v_sw_err=0.0, label=""):
        return cls(
            v_sw=as_kms(v_sw),
            r_obs=float(r_au) * AU_IN_RSUN,
            lon_obs=float(lon_obs),
            lat_obs=float(lat_obs),
            v_sw_err=as_kms(v_sw_err),
            label=label,
        )

    @classmethod
    def near_earth(cls, v_sw, lon_obs, lat_obs=0.0, v_sw_err=0.0, label="OMNI/L1"):
        """A 1 AU (near-Earth) stream; latitude defaults to the ecliptic (~0)."""
        return cls.from_au(v_sw, 1.0, lon_obs, lat_obs, v_sw_err, label)


def from_omni(*args, **kwargs):  # pragma: no cover - optional integration
    """Placeholder for an OMNI/CDAWeb loader (optional 'insitu' extra).

    Intentionally a thin hook: real fetching depends on network + pandas/sunpy and
    is wired up in the worked example, not the importable core.
    """
    raise NotImplementedError(
        "from_omni is provided by the worked example / 'insitu' extra; "
        "construct InSituStream directly for the core API."
    )
