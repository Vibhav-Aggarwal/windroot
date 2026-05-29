import numpy as np
import pytest

from windroot._units import AU_IN_RSUN, RSUN_KM
from windroot.ballistic import (
    ballistic_backmap,
    ballistic_backmap_mc,
    corotation_shift,
)


def test_corotation_shift_matches_hand_calculation():
    # 400 km/s from r_ss=2.5 out to 1 AU, sidereal period 25.38 d.
    v, r_obs, r_ss = 400.0, AU_IN_RSUN, 2.5
    dist_km = (r_obs - r_ss) * RSUN_KM
    travel_s = dist_km / v
    omega_deg_s = 360.0 / (25.38 * 86400.0)
    expected = omega_deg_s * travel_s
    assert corotation_shift(v, r_obs, r_ss=r_ss) == pytest.approx(expected, rel=1e-10)


def test_faster_wind_has_smaller_shift():
    fast = corotation_shift(700.0, AU_IN_RSUN)
    slow = corotation_shift(300.0, AU_IN_RSUN)
    assert slow > fast > 0


def test_backmap_shifts_longitude_forward_and_conserves_latitude():
    res = ballistic_backmap(400.0, AU_IN_RSUN, lon_obs=100.0, lat_obs=7.0)
    assert res.lat_ss == 7.0                      # latitude conserved
    assert res.delta_lon > 0                      # source is ahead in Carrington lon
    assert res.lon_ss == pytest.approx((100.0 + res.delta_lon) % 360.0)
    # slow wind ~ 50-70 deg connection longitude at 1 AU
    assert 40.0 < res.delta_lon < 90.0


def test_mc_ensemble_brackets_deterministic_value():
    rng = np.random.default_rng(0)
    det = ballistic_backmap(450.0, AU_IN_RSUN, 200.0, -5.0, r_ss=2.5)
    ens = ballistic_backmap_mc(
        450.0, AU_IN_RSUN, 200.0, -5.0, v_sw_err=40.0, r_ss_range=(1.5, 3.5), n=5000, rng=rng
    )
    summ = ens.summary()
    assert summ["lat_ss_mean"] == pytest.approx(-5.0, abs=1e-9)  # no lat error
    # circular-mean longitude near the deterministic mid-range value
    assert abs((summ["lon_ss_mean"] - det.lon_ss + 180) % 360 - 180) < 20
    assert summ["lon_ss_std"] > 0
    assert ens.lon_ss.shape == (5000,)
    assert ens.lat_ss.shape == (5000,)


def test_mc_speed_spread_widens_longitude_spread():
    rng = np.random.default_rng(1)
    narrow = ballistic_backmap_mc(450.0, AU_IN_RSUN, 0.0, 0.0, v_sw_err=10.0, n=4000, rng=rng).summary()
    wide = ballistic_backmap_mc(450.0, AU_IN_RSUN, 0.0, 0.0, v_sw_err=80.0, n=4000, rng=rng).summary()
    assert wide["lon_ss_std"] > narrow["lon_ss_std"]
