import numpy as np
import pytest

from windroot.spectro import DopplerMap, _angular_distance


def _demo_map():
    lon = np.arange(0, 360, 2.0)
    lat = np.arange(-60, 61, 2.0)
    upflow = np.zeros((lat.size, lon.size))
    # a bright outflow patch near (lon=180, lat=20)
    LON, LAT = np.meshgrid(lon, lat)
    upflow += 25.0 * np.exp(-(((LON - 180) / 8) ** 2 + ((LAT - 20) / 8) ** 2))
    return DopplerMap(lon=lon, lat=lat, upflow=upflow, line="Ne VIII 770")


def test_shape_validation():
    with pytest.raises(ValueError):
        DopplerMap(lon=np.arange(10), lat=np.arange(5), upflow=np.zeros((4, 9)))


def test_from_vlos_blueshift_is_outflow():
    lon = np.array([0.0, 10.0])
    lat = np.array([0.0])
    vlos = np.array([[-15.0, 5.0]])  # blueshift then redshift
    dm = DopplerMap.from_vlos(lon, lat, vlos)
    assert dm.upflow[0, 0] == 15.0   # blueshift -> positive outflow
    assert dm.upflow[0, 1] == -5.0


def test_sample_finds_outflow_patch():
    dm = _demo_map()
    on_patch = dm.sample(180.0, 20.0, radius_deg=5.0)
    off_patch = dm.sample(0.0, -40.0, radius_deg=5.0)
    assert on_patch > 15.0
    assert abs(off_patch) < 1.0


def test_sample_outside_grid_returns_nan():
    dm = _demo_map()
    assert np.isnan(dm.sample(180.0, 85.0, radius_deg=1.0))  # beyond lat grid


def test_angular_distance_basic():
    d = _angular_distance(0.0, 0.0, np.array([0.0, 90.0]), np.array([0.0, 0.0]))
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    assert d[1] == pytest.approx(90.0, abs=1e-6)
