"""Backtest B — cross-validate windroot's analytic dipole footprinting against an
independent numerical PFSS solution (sunkit-magex) on the same dipolar magnetogram.

If the two implementations agree across a grid of seed locations, the windroot
connectivity math is sound (and the analytic dipole can stand in safely for
quick demos / unit tests).
"""
from __future__ import annotations

import sys

import numpy as np

from backtests.pfss_adapter import PFSSAdapter, synthetic_dipole_synoptic
from windroot.pfss import DipoleSourceSurface


def main(r_ss: float = 2.5, nrho: int = 50) -> int:
    print(f"Backtest B - dipole magnetogram + sunkit-magex PFSS vs windroot analytic dipole")
    print(f"R_ss = {r_ss}, nrho = {nrho}")
    print()

    from sunkit_magex.pfss import Input, pfss

    gong_like = synthetic_dipole_synoptic(b_polar=10.0, nlon=180, nlat=90)
    pfss_out = pfss(Input(gong_like, nrho, r_ss))

    adapter = PFSSAdapter(pfss_out=pfss_out, r_ss=r_ss)
    analytic = DipoleSourceSurface(r_ss=r_ss)

    # Test grid: a range of source-surface footpoints, north + south, near and
    # far from the equator (the streamer-belt singularity).
    lon_ss = np.linspace(10, 350, 8)
    lat_ss = np.array([-60, -40, -20, -5, 5, 20, 40, 60], dtype=float)

    print("| lon_ss | lat_ss | windroot (lon, lat) | sunkit-magex (lon, lat) | dlon | dlat |")
    print("|-------:|-------:|---------------------|--------------------------|-----:|-----:|")

    rows = []
    for ls in lat_ss:
        # vectorised on the longitude axis at a fixed lat
        loa, laa = analytic.map_footpoints(lon_ss, np.full_like(lon_ss, ls))
        lop, lap = adapter.map_footpoints(lon_ss, np.full_like(lon_ss, ls))
        for i in range(lon_ss.size):
            if np.isnan(lop[i]):
                continue
            dlon = (loa[i] - lop[i] + 180) % 360 - 180
            dlat = laa[i] - lap[i]
            rows.append((lon_ss[i], ls, loa[i], laa[i], lop[i], lap[i], dlon, dlat))
            print(f"| {lon_ss[i]:6.1f} | {ls:6.1f} "
                  f"| ({loa[i]:6.1f}, {laa[i]:6.1f}) "
                  f"| ({lop[i]:6.1f}, {lap[i]:6.1f}) "
                  f"| {dlon:5.2f} | {dlat:5.2f} |")

    if not rows:
        print("\nNo successful traces - tracer failed everywhere.")
        return 1

    dlon = np.array([r[6] for r in rows])
    dlat = np.array([r[7] for r in rows])

    rms_lon = float(np.sqrt(np.mean(dlon ** 2)))
    rms_lat = float(np.sqrt(np.mean(dlat ** 2)))
    print(f"\nRMS disagreement: dlon = {rms_lon:.2f} deg, dlat = {rms_lat:.2f} deg "
          f"across {len(rows)} seeds.")

    # The dipole magnetogram + axisymmetric PFSS should agree well; the
    # numeric tracer + finite grid resolution + WCS round-trip set the floor.
    ok = rms_lon < 3.0 and rms_lat < 5.0
    print(f"\nAGREEMENT: {'PASS' if ok else 'FAIL'} (tol: dlon<3 deg, dlat<5 deg)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
