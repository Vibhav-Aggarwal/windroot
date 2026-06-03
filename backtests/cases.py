"""Backtest case registry.

A :class:`BacktestCase` describes one event:
- spacecraft pose and measured solar wind speed (the *input*),
- the *expected* photospheric source-region footpoint and a tolerance,
- the source reference (paper / dataset citation),
- optional Carrington rotation + magnetogram hint for the real-data backtest.

Synthetic cases set ``carrington_rotation=None`` and use the analytic dipole.
Real cases name a Carrington rotation; the runner fetches the matching GONG
synoptic magnetogram and builds a PFSS solution via sunkit-magex.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestCase:
    case_id: str
    label: str
    reference: str

    # spacecraft pose at the in-situ measurement (Carrington)
    r_obs_rsun: float
    lon_obs_deg: float
    lat_obs_deg: float

    # measured radial solar-wind speed
    v_sw_kms: float
    v_sw_err_kms: float = 30.0

    # expected photospheric source footpoint (Carrington)
    expected_lon_deg: float = 0.0
    expected_lat_deg: float = 0.0
    tolerance_deg: float = 15.0   # great-circle radius for "pass"

    # which field to use
    backend: str = "dipole"       # "dipole" or "pfss"
    carrington_rotation: Optional[int] = None
    magnetogram_provider: str = "GONG"
    r_ss_rsun: float = 2.5

    notes: str = ""


# -----------------------------------------------------------------------------
# Synthetic test cases — ground truth = analytic dipole footpoint at the
# back-mapped source-surface longitude (lat depends on r_ss). We seed
# expected_lat to the dipole's mapping; for these cases we mostly check that
# the back-map + dipole pipeline puts the top candidate within tolerance of
# the truth.
# -----------------------------------------------------------------------------
def _synthetic_cases() -> list[BacktestCase]:
    cases: list[BacktestCase] = []
    # vary v_sw -> different longitudinal shifts
    for sid, v_sw in enumerate([320.0, 420.0, 550.0, 700.0]):
        cases.append(BacktestCase(
            case_id=f"SYN-VSW-{sid}",
            label=f"synthetic slow/fast wind, v={v_sw:.0f} km/s",
            reference="windroot synthetic (analytic dipole)",
            r_obs_rsun=215.0, lon_obs_deg=120.0, lat_obs_deg=3.0,
            v_sw_kms=v_sw, v_sw_err_kms=25.0,
            # expected longitude = lon_obs + corotation shift (filled by runner if NaN)
            expected_lon_deg=float("nan"),
            expected_lat_deg=float("nan"),   # filled from dipole truth at runtime
            tolerance_deg=10.0,
            backend="dipole",
            notes="varying wind speed; tests longitudinal-shift correctness",
        ))
    # vary spacecraft latitude
    for sid, lat in enumerate([-30.0, -10.0, 0.0, 15.0, 35.0]):
        cases.append(BacktestCase(
            case_id=f"SYN-LAT-{sid}",
            label=f"synthetic ecliptic-off, lat={lat:+.0f} deg",
            reference="windroot synthetic (analytic dipole)",
            r_obs_rsun=215.0, lon_obs_deg=90.0, lat_obs_deg=lat,
            v_sw_kms=450.0, v_sw_err_kms=25.0,
            expected_lon_deg=float("nan"),
            expected_lat_deg=float("nan"),
            tolerance_deg=10.0,
            backend="dipole",
            notes="varying spacecraft latitude; tests latitude conservation + dipole fan",
        ))
    return cases


# -----------------------------------------------------------------------------
# Real-data cases (require [pfss] extra + network for GONG synoptics).
# Sources: published PSP / Solar Orbiter encounters with identified footpoints.
# Numbers are conservative tolerances given known PFSS+ballistic uncertainties.
# -----------------------------------------------------------------------------
def _real_cases() -> list[BacktestCase]:
    return [
        BacktestCase(
            case_id="PSP-E1",
            label="PSP Encounter 1 perihelion (2018-11-06, CR 2210)",
            reference="Bale et al. 2019 (Nature); Badman et al. 2020 (ApJS)",
            r_obs_rsun=35.7,            # ~0.166 AU
            lon_obs_deg=158.0,
            lat_obs_deg=-3.5,
            v_sw_kms=340.0, v_sw_err_kms=40.0,
            # Bale 2019: small equatorial CH near central meridian; Badman E1 footpoint
            expected_lon_deg=160.0,
            expected_lat_deg=-5.0,
            tolerance_deg=20.0,
            backend="pfss",
            carrington_rotation=2210,
            notes="E1 perihelion source identified as small near-equatorial CH",
        ),
        BacktestCase(
            case_id="PSP-E4",
            label="PSP Encounter 4 perihelion (2020-01-29, CR 2226)",
            reference="Panasenco et al. 2020 (ApJS)",
            r_obs_rsun=27.8,            # ~0.13 AU
            lon_obs_deg=205.0,
            lat_obs_deg=-3.8,
            v_sw_kms=320.0, v_sw_err_kms=35.0,
            expected_lon_deg=210.0,
            expected_lat_deg=-15.0,
            tolerance_deg=25.0,
            backend="pfss",
            carrington_rotation=2226,
            notes="slow-wind streamer-belt source; tolerance widened",
        ),
        BacktestCase(
            case_id="OMNI-CH-HSS",
            label="L1 high-speed stream from a polar coronal hole extension",
            reference="Karachik & Pevtsov 2011 (Solar Phys); illustrative",
            r_obs_rsun=215.0, lon_obs_deg=45.0, lat_obs_deg=3.0,
            v_sw_kms=650.0, v_sw_err_kms=30.0,
            expected_lon_deg=110.0,     # rough estimate; refined when CR known
            expected_lat_deg=40.0,      # mid-latitude coronal hole boundary
            tolerance_deg=25.0,
            backend="pfss",
            carrington_rotation=2210,   # placeholder; not the actual CR
            notes="illustrative HSS test; CR + footpoint refined when fetched",
        ),
    ]


def all_cases(include_real: bool = True) -> list[BacktestCase]:
    cases = _synthetic_cases()
    if include_real:
        cases += _real_cases()
    return cases
