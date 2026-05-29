import numpy as np
import pytest

from windroot import DopplerMap, InSituStream, find_sources
from windroot.pfss import DipoleSourceSurface


def test_end_to_end_runs_with_no_external_data():
    stream = InSituStream.near_earth(v_sw=420, lon_obs=120, lat_obs=3, v_sw_err=30)
    result = find_sources(stream, n_mc=3000, rng=np.random.default_rng(42))
    assert result.candidates
    assert result.best.confidence > 0
    assert result.lon_phot.shape == result.lat_phot.shape == (3000,)
    # table renders
    assert "rank" in result.to_table()


def test_faster_wind_maps_closer_to_spacecraft_longitude():
    rng = np.random.default_rng(0)
    fast = find_sources(InSituStream.near_earth(700, 100, 0, 20), n_mc=4000, rng=rng)
    slow = find_sources(InSituStream.near_earth(320, 100, 0, 20), n_mc=4000, rng=rng)
    # source-surface footpoint longitude shift grows for slower wind
    assert slow.ensemble.summary()["lon_ss_mean"] > fast.ensemble.summary()["lon_ss_mean"]


def test_spectroscopic_layer_changes_best_candidate():
    rng = np.random.default_rng(7)
    stream = InSituStream.near_earth(v_sw=400, lon_obs=120, lat_obs=2, v_sw_err=40)

    base = find_sources(stream, n_mc=4000, rng=np.random.default_rng(7))
    # place an outflow patch on a *secondary* candidate, away from the top one
    if len(base.candidates) < 2:
        pytest.skip("need >=2 candidates to demonstrate re-ranking")
    secondary = base.candidates[1]

    glon = np.arange(0, 360, 2.0)
    glat = np.arange(-90, 91, 2.0)
    LON, LAT = np.meshgrid(glon, glat)
    upflow = 35.0 * np.exp(
        -(((LON - secondary.lon) / 6) ** 2 + ((LAT - secondary.lat) / 6) ** 2)
    )
    dm = DopplerMap(glon, glat, upflow, line="Ne VIII 770")

    fused = find_sources(
        stream, doppler_map=dm, n_mc=4000, rng=np.random.default_rng(7),
        w_connectivity=0.35, w_outflow=0.65,
    )
    assert fused.best is not None
    # the fused winner should now sit on the outflow patch (the secondary location)
    assert fused.best.lat == pytest.approx(secondary.lat, abs=6)
    assert fused.best.outflow_kms > 10.0


def test_custom_field_is_used():
    stream = InSituStream.near_earth(v_sw=450, lon_obs=80, lat_obs=10, v_sw_err=25)
    field = DipoleSourceSurface(r_ss=2.0)
    result = find_sources(stream, field=field, n_mc=2000, rng=np.random.default_rng(1))
    assert result.candidates
