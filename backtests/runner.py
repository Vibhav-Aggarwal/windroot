"""Backtest runner — execute a list of :class:`BacktestCase`s and report results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from windroot import InSituStream, find_sources
from windroot.ballistic import corotation_shift
from windroot.pfss import DipoleSourceSurface
from windroot.spectro import DopplerMap

from .cases import BacktestCase


def angular_distance_deg(lon1, lat1, lon2, lat2) -> float:
    """Great-circle angular distance in degrees."""
    a = np.deg2rad([lon1, lat1])
    b = np.deg2rad([lon2, lat2])
    cd = np.sin(a[1]) * np.sin(b[1]) + np.cos(a[1]) * np.cos(b[1]) * np.cos(a[0] - b[0])
    return float(np.rad2deg(np.arccos(np.clip(cd, -1.0, 1.0))))


@dataclass
class BacktestResult:
    case_id: str
    label: str
    expected_lon: float
    expected_lat: float
    best_lon: float
    best_lat: float
    best_conf: float
    error_deg: float
    tolerance_deg: float
    passed: bool
    backend: str
    n_candidates: int
    notes: str = ""


def _fill_synthetic_truth(case: BacktestCase, r_ss: float) -> tuple[float, float]:
    """Compute the deterministic dipole-truth footpoint for a synthetic case."""
    field = DipoleSourceSurface(r_ss=r_ss)
    dlon = corotation_shift(case.v_sw_kms, case.r_obs_rsun, r_ss=r_ss)
    lon_ss = (case.lon_obs_deg + dlon) % 360.0
    fp = field.footpoint(lon_ss=lon_ss, lat_ss=case.lat_obs_deg)
    return fp.lon, fp.lat


def run_case(
    case: BacktestCase,
    field_factory: Optional[Callable[[BacktestCase], object]] = None,
    doppler_map: Optional[DopplerMap] = None,
    n_mc: int = 6000,
    rng: Optional[np.random.Generator] = None,
) -> BacktestResult:
    """Run a single backtest case.

    ``field_factory`` builds the magnetic-field mapper for the case (``None``
    falls back to the analytic dipole at ``case.r_ss_rsun``). Real-data cases
    pass a factory that constructs a sunkit-magex PFSS wrapper.
    """
    rng = rng or np.random.default_rng(42)
    field = field_factory(case) if field_factory else DipoleSourceSurface(r_ss=case.r_ss_rsun)

    # for synthetic cases, fill the expected footpoint from the dipole truth
    if case.backend == "dipole" and (
        np.isnan(case.expected_lon_deg) or np.isnan(case.expected_lat_deg)
    ):
        exp_lon, exp_lat = _fill_synthetic_truth(case, r_ss=case.r_ss_rsun)
    else:
        exp_lon, exp_lat = case.expected_lon_deg, case.expected_lat_deg

    stream = InSituStream(
        v_sw=case.v_sw_kms,
        r_obs=case.r_obs_rsun,
        lon_obs=case.lon_obs_deg,
        lat_obs=case.lat_obs_deg,
        v_sw_err=case.v_sw_err_kms,
        label=case.case_id,
    )
    result = find_sources(
        stream, field=field, doppler_map=doppler_map, n_mc=n_mc, rng=rng,
        r_ss_range=(max(1.2, case.r_ss_rsun - 0.5), case.r_ss_rsun + 0.5),
    )

    if not result.candidates:
        return BacktestResult(
            case_id=case.case_id, label=case.label,
            expected_lon=exp_lon, expected_lat=exp_lat,
            best_lon=float("nan"), best_lat=float("nan"), best_conf=0.0,
            error_deg=float("nan"), tolerance_deg=case.tolerance_deg,
            passed=False, backend=case.backend, n_candidates=0,
            notes="no candidates produced",
        )

    best = result.candidates[0]
    err = angular_distance_deg(best.lon, best.lat, exp_lon, exp_lat)
    return BacktestResult(
        case_id=case.case_id, label=case.label,
        expected_lon=exp_lon, expected_lat=exp_lat,
        best_lon=best.lon, best_lat=best.lat, best_conf=best.confidence,
        error_deg=err, tolerance_deg=case.tolerance_deg,
        passed=err <= case.tolerance_deg,
        backend=case.backend, n_candidates=len(result.candidates),
        notes=case.notes,
    )


def run_all(
    cases: list[BacktestCase],
    field_factory: Optional[Callable[[BacktestCase], object]] = None,
    doppler_map_factory: Optional[Callable[[BacktestCase], DopplerMap]] = None,
    n_mc: int = 6000,
    seed: int = 42,
) -> list[BacktestResult]:
    results: list[BacktestResult] = []
    for i, case in enumerate(cases):
        dm = doppler_map_factory(case) if doppler_map_factory else None
        results.append(run_case(case, field_factory=field_factory, doppler_map=dm,
                                n_mc=n_mc, rng=np.random.default_rng(seed + i)))
    return results


def format_table(results: list[BacktestResult]) -> str:
    """Render results as a markdown-friendly table."""
    lines = [
        "| case | backend | expected (lon, lat) | best (lon, lat) | err [deg] | tol | pass |",
        "|------|---------|---------------------|------------------|----------:|----:|------|",
    ]
    for r in results:
        pass_str = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| {r.case_id} | {r.backend} | "
            f"({r.expected_lon:.1f}, {r.expected_lat:.1f}) | "
            f"({r.best_lon:.1f}, {r.best_lat:.1f}) | "
            f"{r.error_deg:8.2f} | {r.tolerance_deg:3.0f} | {pass_str} |"
        )
    passed = sum(1 for r in results if r.passed)
    lines.append(f"\n**{passed}/{len(results)} cases passed.**")
    return "\n".join(lines)
