"""Backtest C — full pipeline on a REAL GONG synoptic magnetogram.

For multiple synthetic in-situ streams across the Carrington longitude grid we:
1. solve PFSS on the real CR 2234 GONG magnetogram (via sunkit-magex),
2. run windroot.find_sources with the PFSSAdapter so the connectivity uses
   the real (non-axisymmetric) field,
3. confirm the pipeline produces ranked candidates with valid footpoints,
4. compare against the analytic-dipole result for the same streams — the two
   must differ (the real magnetogram is not a dipole), which itself proves
   PFSS is delivering structure-aware mapping rather than echoing the dipole.
"""
from __future__ import annotations

import sys

import numpy as np

from backtests.pfss_adapter import PFSSAdapter
from windroot import InSituStream, find_sources
from windroot.pfss import DipoleSourceSurface


def _load_gong_pfss(r_ss: float = 2.5, nrho: int = 35):
    """Fetch + load the bundled GONG sample and solve PFSS."""
    import sunpy.map
    from sunkit_magex.pfss import Input, pfss
    from sunkit_magex.pfss.sample_data import get_gong_map

    gong = sunpy.map.Map(str(get_gong_map()))
    out = pfss(Input(gong, nrho, r_ss))
    return gong, out


def angular_distance(lon1, lat1, lon2, lat2) -> float:
    a = np.deg2rad([lon1, lat1]); b = np.deg2rad([lon2, lat2])
    cd = np.sin(a[1])*np.sin(b[1]) + np.cos(a[1])*np.cos(b[1])*np.cos(a[0]-b[0])
    return float(np.rad2deg(np.arccos(np.clip(cd, -1.0, 1.0))))


def main() -> int:
    r_ss = 2.5
    print(f"Backtest C - real GONG synoptic magnetogram (CR 2234, 2020-09-01)")
    print(f"R_ss = {r_ss}, multiple in-situ streams across Carrington longitudes")
    print()

    print("Solving PFSS on real GONG magnetogram (sunkit-magex)... ", end="", flush=True)
    gong, pfss_out = _load_gong_pfss(r_ss=r_ss, nrho=35)
    print("ok")
    print(f"  magnetogram shape: {gong.data.shape}, B_r range "
          f"[{gong.data.min():.1f}, {gong.data.max():.1f}] G")

    pfss_field = PFSSAdapter(pfss_out=pfss_out, r_ss=r_ss)
    dipole_field = DipoleSourceSurface(r_ss=r_ss)

    # 6 streams spanning Carrington longitudes 0-300, ecliptic-ish, slow & fast wind.
    streams = []
    for lon, v in zip([30, 90, 150, 210, 270, 330], [380, 520, 360, 640, 410, 480]):
        streams.append(InSituStream.near_earth(
            v_sw=v, lon_obs=lon, lat_obs=3.0, v_sw_err=30.0,
            label=f"lon={lon} v={v}",
        ))

    print()
    print("| stream | v_sw | dipole footpoint (lon, lat) | PFSS footpoint (lon, lat) | "
          "delta [deg] | PFSS conf |")
    print("|--------|-----:|-----------------------------|----------------------------|"
          "-----------:|----------:|")
    diffs = []
    failures = 0
    for s in streams:
        rd = find_sources(s, field=dipole_field, n_mc=4000, rng=np.random.default_rng(11))
        rp = find_sources(s, field=pfss_field, n_mc=600, rng=np.random.default_rng(11))
        # PFSS path is slow; smaller MC. Skip if no candidates.
        if not rp.candidates:
            failures += 1
            print(f"| {s.label} | {s.v_sw:4.0f} | "
                  f"({rd.best.lon:5.1f}, {rd.best.lat:5.1f}) | "
                  f"(no PFSS candidate; all seeds closed?) | -- | -- |")
            continue
        d = angular_distance(rd.best.lon, rd.best.lat, rp.best.lon, rp.best.lat)
        diffs.append(d)
        print(f"| {s.label} | {s.v_sw:4.0f} | "
              f"({rd.best.lon:5.1f}, {rd.best.lat:5.1f}) | "
              f"({rp.best.lon:5.1f}, {rp.best.lat:5.1f}) | "
              f"{d:8.1f} | {rp.best.confidence:.2f} |")

    if not diffs:
        print("\nFAIL: PFSS produced no candidates for any stream.")
        return 1

    median_diff = float(np.median(diffs))
    print(f"\n{len(streams) - failures}/{len(streams)} streams produced a PFSS candidate.")
    print(f"Median dipole-vs-PFSS footpoint separation: {median_diff:.1f} deg")
    print("(non-zero separation confirms PFSS is using real magnetogram structure, "
          "not echoing the analytic dipole.)")

    ok = (len(streams) - failures) >= max(1, len(streams) // 2) and median_diff > 5.0
    print(f"\nPIPELINE: {'PASS' if ok else 'FAIL'}")
    print("(criteria: >=half of streams produce candidates, AND median separation "
          ">5 deg from the dipole result)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
