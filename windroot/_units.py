"""Shared constants, units, and small coordinate helpers.

Conventions used throughout windroot:
- Longitudes/latitudes are Carrington heliographic coordinates in **degrees**.
- Radii are in **solar radii** (Rsun) unless a Quantity says otherwise.
- Speeds are in **km/s**.
- "Outflow" / "upflow" velocity is **positive outward** (a blueshift on the disk).
"""
from __future__ import annotations

import astropy.units as u
from astropy.constants import R_sun, au

#: Solar sidereal rotation period (Carrington). Used for ballistic corotation.
SIDEREAL_ROTATION_PERIOD = 25.38 * u.day

#: Default Schatten/Hoeksema source-surface radius for PFSS.
DEFAULT_R_SS = 2.5  # Rsun

#: 1 AU expressed in solar radii (~215).
AU_IN_RSUN = (au / R_sun).decompose().value

#: Solar radius in km.
RSUN_KM = R_sun.to(u.km).value


def as_rsun(r) -> float:
    """Coerce a radius to a float in solar radii.

    Accepts a plain number (already Rsun) or an astropy length Quantity.
    """
    if isinstance(r, u.Quantity):
        return (r / R_sun).decompose().value
    return float(r)


def as_kms(v) -> float:
    """Coerce a speed to a float in km/s."""
    if isinstance(v, u.Quantity):
        return v.to(u.km / u.s).value
    return float(v)


def sidereal_rate_deg_per_s(period=SIDEREAL_ROTATION_PERIOD) -> float:
    """Solar rotation rate in degrees per second for the given sidereal period."""
    return 360.0 / period.to(u.s).value


def wrap_longitude(lon_deg: float) -> float:
    """Wrap a longitude into [0, 360)."""
    return float(lon_deg) % 360.0
