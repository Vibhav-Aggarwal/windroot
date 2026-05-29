import numpy as np
import pytest

from windroot.pfss import DipoleSourceSurface, map_footpoints


@pytest.mark.parametrize("lat_ss", [5.0, 15.0, 30.0, 60.0, -10.0, -45.0])
def test_numeric_tracer_matches_analytic_footpoint(lat_ss):
    """The independent dtheta/dr integration must reproduce the closed-form mapping.

    This is the synthetic 'known footpoint recovery' validation.
    """
    field = DipoleSourceSurface(r_ss=2.5)
    analytic = field.footpoint(lon_ss=123.0, lat_ss=lat_ss)
    numeric = field.trace_footpoint(lon_ss=123.0, lat_ss=lat_ss)
    assert numeric.lat == pytest.approx(analytic.lat, abs=0.2)
    assert numeric.lon == analytic.lon == 123.0


def test_field_line_invariant_is_conserved():
    """(r^2 + 2 R_ss^3 / r) sin^2(theta) is constant along the traced line."""
    field = DipoleSourceSurface(r_ss=2.5)
    lat_ss = 20.0
    theta_ss = np.deg2rad(90 - lat_ss)
    inv_ss = (field.r_ss**2 + 2 * field.r_ss**3 / field.r_ss) * np.sin(theta_ss) ** 2

    fp = field.footpoint(123.0, lat_ss)
    theta_phot = np.deg2rad(90 - fp.lat)
    inv_phot = (field.r_sun**2 + 2 * field.r_ss**3 / field.r_sun) * np.sin(theta_phot) ** 2
    assert inv_phot == pytest.approx(inv_ss, rel=1e-6)


def test_source_surface_equator_maps_to_open_field_boundary():
    """A near-equatorial source-surface point maps to ~40 deg (polar-hole edge)."""
    field = DipoleSourceSurface(r_ss=2.5)
    fp = field.footpoint(0.0, 0.5)
    assert 35.0 < abs(fp.lat) < 45.0


def test_higher_source_surface_pushes_footpoint_poleward():
    near_eq = 2.0
    fp_low = DipoleSourceSurface(r_ss=2.0).footpoint(0.0, near_eq)
    fp_high = DipoleSourceSurface(r_ss=3.0).footpoint(0.0, near_eq)
    assert abs(fp_high.lat) > abs(fp_low.lat)


def test_map_footpoints_vectorised_matches_scalar():
    field = DipoleSourceSurface(r_ss=2.5)
    lons = np.array([10.0, 200.0, 350.0])
    lats = np.array([5.0, -20.0, 40.0])
    vlon, vlat = map_footpoints(field, lons, lats)
    for i in range(3):
        fp = field.footpoint(lons[i], lats[i])
        assert vlon[i] == pytest.approx(fp.lon)
        assert vlat[i] == pytest.approx(fp.lat, abs=1e-9)


def test_hemisphere_is_preserved():
    field = DipoleSourceSurface(r_ss=2.5)
    assert field.footpoint(0.0, 25.0).lat > 0
    assert field.footpoint(0.0, -25.0).lat < 0
