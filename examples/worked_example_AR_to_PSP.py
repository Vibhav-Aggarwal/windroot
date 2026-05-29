"""Worked example: back-map a solar-wind stream and add spectroscopic evidence.

Scenario (illustrative, runs with the core only — no network/heavy deps):
a slow (~360 km/s) stream observed in the ecliptic is back-mapped to the Sun.
Magnetic connectivity alone yields several candidate source regions; adding a
SPICE/EIS-like Doppler *outflow* map promotes the candidate where plasma is
actually seen flowing outward — the windroot contribution.

To run against REAL data, install the extras and swap in the marked blocks:

    pip install 'windroot[pfss,data,insitu,plot]'

    # real magnetogram + PFSS field (instead of the analytic dipole):
    import sunpy.map
    from windroot.pfss import pfss_connectivity
    gong = sunpy.map.Map("path/to/gong_synoptic.fits")
    pfss_out = pfss_connectivity(gong, r_ss=2.5)         # then trace footpoints
    #   (wire pfss_out tracing into a field adapter exposing .footpoint())

    # real in-situ stream (PSP/OMNI) -> InSituStream(v_sw, r_obs, lon_obs, lat_obs)
    # real SPICE Doppler map -> DopplerMap.from_vlos(lon, lat, vlos)  (optionally
    #   pre-cleaned with windroot.spectro.apply_skew_correction, Arpit's A&A 706 method)
"""
import numpy as np

from windroot import DopplerMap, InSituStream, find_sources

OUT_PNG = "windroot_sources.png"


def synthetic_outflow_map(centers):
    """A SPICE/EIS-like outflow map: blueshifted upflow patches at given centres.

    centers : list of (lon, lat, amplitude_km_s)
    """
    lon = np.arange(0, 360, 1.0)
    lat = np.arange(-70, 71, 1.0)
    LON, LAT = np.meshgrid(lon, lat)
    upflow = np.zeros_like(LON)
    for clon, clat, amp in centers:
        # wrap-aware longitude distance
        dlon = (LON - clon + 180) % 360 - 180
        upflow += amp * np.exp(-((dlon / 7.0) ** 2 + ((LAT - clat) / 7.0) ** 2))
    # mild quiet-Sun noise
    upflow += 1.0 * np.random.default_rng(0).standard_normal(upflow.shape)
    return DopplerMap(lon, lat, upflow, line="Ne VIII 770")


def main():
    rng = np.random.default_rng(2024)

    # A slow solar-wind stream observed near Earth (1 AU), ecliptic.
    stream = InSituStream.near_earth(v_sw=360, lon_obs=140, lat_obs=4, v_sw_err=35,
                                     label="slow stream @ L1 (illustrative)")

    print("=" * 68)
    print("STEP 1 - magnetic connectivity only (classical)")
    print("=" * 68)
    base = find_sources(stream, n_mc=6000, rng=rng)
    ens = base.ensemble.summary()
    print(f"source-surface footpoint: lon = {ens['lon_ss_mean']:.1f} "
          f"+/- {ens['lon_ss_std']:.1f} deg,  lat = {ens['lat_ss_mean']:.1f} deg")
    print(base.to_table())

    if len(base.candidates) < 2:
        print("\n(only one connectivity candidate; widen errors to see more)")
        return

    top = base.candidates[0]
    secondary = base.candidates[1]

    # SPICE/EIS shows strong outflow at the *secondary* connectivity candidate
    # (e.g. an active-region boundary upflow) and little at the top one.
    print("\n" + "=" * 68)
    print("STEP 2 - add spectroscopic outflow evidence (windroot)")
    print("=" * 68)
    dm = synthetic_outflow_map([
        (secondary.lon, secondary.lat, 28.0),   # real upflow here
        (top.lon, top.lat, 3.0),                 # negligible at the magnetic-best
    ])
    fused = find_sources(stream, doppler_map=dm, n_mc=6000,
                         rng=np.random.default_rng(2024),
                         w_connectivity=0.4, w_outflow=0.6)
    print(fused.to_table())

    print("\nresult:")
    print(f"  connectivity-only best : lon={top.lon:.0f}, lat={top.lat:.0f}")
    print(f"  fused best             : lon={fused.best.lon:.0f}, lat={fused.best.lat:.0f}, "
          f"outflow={fused.best.outflow_kms:.1f} km/s")
    print("  -> spectroscopic evidence re-ranked the source to where plasma "
          "actually flows outward.")

    try:
        from windroot.viz import plot_sources
        plot_sources(fused, doppler_map=dm, savepath=OUT_PNG)
        print(f"\nfigure saved: {OUT_PNG}")
    except ImportError:
        print("\n(install 'windroot[plot]' for the figure)")


if __name__ == "__main__":
    main()
