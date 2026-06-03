"""Backtest A — synthetic multi-case (analytic-dipole known truth).

Runs all synthetic cases through windroot end-to-end and verifies the top
candidate from the Monte-Carlo + binning pipeline recovers the deterministic
dipole truth within tolerance. Also runs each case again with an injected
outflow patch off the true footpoint to verify the spectroscopic layer can
shift the top-candidate position.
"""
from __future__ import annotations

import sys

import numpy as np

from backtests.cases import _synthetic_cases
from backtests.runner import format_table, run_all
from windroot.spectro import DopplerMap


def _outflow_at(lon: float, lat: float, amp: float = 30.0) -> DopplerMap:
    lon_g = np.arange(0, 360, 1.0)
    lat_g = np.arange(-70, 71, 1.0)
    LON, LAT = np.meshgrid(lon_g, lat_g)
    dlon = (LON - lon + 180) % 360 - 180
    upflow = amp * np.exp(-((dlon / 8.0) ** 2 + ((LAT - lat) / 8.0) ** 2))
    return DopplerMap(lon_g, lat_g, upflow, line="synthetic")


def main() -> int:
    cases = _synthetic_cases()

    # Pass 1: connectivity-only recovery of the dipole truth
    print("Backtest A.1 — connectivity-only (top candidate vs analytic dipole truth)")
    print()
    res_conn = run_all(cases)
    print(format_table(res_conn))

    # Pass 2: robustness - spurious outflow far from the truth must NOT
    # hijack the top candidate when connectivity is decisive.
    print("\n\nBacktest A.2 — robustness to spurious outflow (60 deg off truth)")
    print()
    hijacked = 0
    for r in res_conn:
        decoy_lon = (r.expected_lon + 60.0) % 360.0
        decoy_lat = r.expected_lat
        dm = _outflow_at(decoy_lon, decoy_lat, amp=40.0)
        res = run_all([next(c for c in cases if c.case_id == r.case_id)],
                      doppler_map_factory=lambda _c, _dm=dm: _dm,
                      n_mc=6000)
        new = res[0]
        hijacked += 0 if new.passed else 1
        flag = "ok" if new.passed else "HIJACKED"
        print(f"  {r.case_id:>10}  decoy at ({decoy_lon:6.1f}, {decoy_lat:5.1f})  "
              f"best ({new.best_lon:6.1f}, {new.best_lat:5.1f})  "
              f"err {new.error_deg:5.1f} deg  {flag}")
    print(f"\n{len(res_conn) - hijacked}/{len(res_conn)} cases robust to a 60 deg "
          f"spurious outflow.")

    # Pass 3: spectroscopic discrimination - widen errors so connectivity is
    # ambiguous, then show outflow at the truth ranks #1.
    print("\n\nBacktest A.3 — spectroscopic discrimination under MC ambiguity")
    print()
    discriminated = 0
    from dataclasses import replace
    for c in cases:
        amb = replace(c, v_sw_err_kms=120.0)  # very wide MC -> several candidates
        res_amb_conn = run_all([amb], n_mc=8000)[0]
        # truth from deterministic dipole
        from backtests.runner import _fill_synthetic_truth
        truth_lon, truth_lat = _fill_synthetic_truth(amb, r_ss=amb.r_ss_rsun)
        dm = _outflow_at(truth_lon, truth_lat, amp=40.0)
        res_amb_fused = run_all([amb], doppler_map_factory=lambda _c, _dm=dm: _dm,
                                n_mc=8000)[0]
        better = res_amb_fused.error_deg <= res_amb_conn.error_deg
        discriminated += 1 if better else 0
        mark = "->" if better else "<-"
        print(f"  {c.case_id:>10}  conn err {res_amb_conn.error_deg:5.1f} deg "
              f"{mark} fused err {res_amb_fused.error_deg:5.1f} deg "
              f"({'discriminated' if better else 'no improvement'})")
    print(f"\n{discriminated}/{len(cases)} cases improved when outflow "
          f"evidence was added under ambiguity.")

    n_pass = sum(1 for r in res_conn if r.passed)
    return 0 if (n_pass == len(res_conn) and hijacked == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
