import numpy as np
import pytest

from windroot.rank import rank_sources
from windroot.spectro import DopplerMap


def _two_clusters(rng):
    """Cluster A (dense, lon~100/lat~20) + cluster B (sparser, lon~250/lat~-10)."""
    a_lon = 100 + 2.0 * rng.standard_normal(700)
    a_lat = 20 + 2.0 * rng.standard_normal(700)
    b_lon = 250 + 2.0 * rng.standard_normal(300)
    b_lat = -10 + 2.0 * rng.standard_normal(300)
    return np.concatenate([a_lon, b_lon]) % 360, np.concatenate([a_lat, b_lat])


def test_connectivity_only_ranks_denser_cluster_first():
    rng = np.random.default_rng(0)
    lon, lat = _two_clusters(rng)
    cands = rank_sources(lon, lat, doppler_map=None)
    assert cands, "expected candidates"
    top = cands[0]
    assert abs((top.lon - 100 + 180) % 360 - 180) < 6   # cluster A
    assert top.lat == pytest.approx(20, abs=4)
    assert all(np.isnan(c.outflow_kms) for c in cands)


def test_spectroscopic_evidence_reranks_to_outflow_region():
    """The novel claim: strong outflow at the *weaker*-connectivity cluster B
    promotes it above the magnetically-denser cluster A."""
    rng = np.random.default_rng(0)
    lon, lat = _two_clusters(rng)

    glon = np.arange(0, 360, 2.0)
    glat = np.arange(-60, 61, 2.0)
    LON, LAT = np.meshgrid(glon, glat)
    # strong outflow only at cluster B (250, -10), nothing at A
    upflow = 30.0 * np.exp(-(((LON - 250) / 7) ** 2 + ((LAT + 10) / 7) ** 2))
    dm = DopplerMap(glon, glat, upflow, line="O VI 1032")

    fused = rank_sources(lon, lat, doppler_map=dm, w_connectivity=0.4, w_outflow=0.6)
    top = fused[0]
    assert abs((top.lon - 250 + 180) % 360 - 180) < 6   # cluster B now wins
    assert top.lat == pytest.approx(-10, abs=4)
    assert top.outflow_kms > 15.0


def test_weights_control_the_balance():
    rng = np.random.default_rng(0)
    lon, lat = _two_clusters(rng)
    glon = np.arange(0, 360, 2.0)
    glat = np.arange(-60, 61, 2.0)
    LON, LAT = np.meshgrid(glon, glat)
    upflow = 30.0 * np.exp(-(((LON - 250) / 7) ** 2 + ((LAT + 10) / 7) ** 2))
    dm = DopplerMap(glon, glat, upflow)

    # pure connectivity weighting must keep dense cluster A on top
    conn_only = rank_sources(lon, lat, doppler_map=dm, w_connectivity=1.0, w_outflow=0.0)
    assert abs((conn_only[0].lon - 100 + 180) % 360 - 180) < 6


def test_empty_cloud_returns_no_candidates():
    assert rank_sources(np.array([]), np.array([])) == []


def test_confidence_in_unit_interval():
    rng = np.random.default_rng(2)
    lon, lat = _two_clusters(rng)
    for c in rank_sources(lon, lat):
        assert 0.0 <= c.confidence <= 1.0
        assert 0.0 <= c.connectivity_score <= 1.0


def test_uncovered_candidate_falls_back_to_connectivity_only():
    """A Doppler map that doesn't spatially cover a candidate must leave it
    scored by connectivity only — not penalised as if it were a confirmed
    downflow. This is the post-review semantic fix.
    """
    import math
    rng = np.random.default_rng(11)
    lon, lat = _two_clusters(rng)
    # Doppler grid covers only cluster B's neighbourhood; cluster A is outside.
    glon = np.arange(240, 261, 1.0)
    glat = np.arange(-20, 1, 1.0)
    LON, LAT = np.meshgrid(glon, glat)
    upflow = 30.0 * np.exp(-(((LON - 250) / 5) ** 2 + ((LAT + 10) / 5) ** 2))
    dm = DopplerMap(glon, glat, upflow)

    cands = rank_sources(lon, lat, doppler_map=dm, w_connectivity=0.5, w_outflow=0.5)
    a = next(c for c in cands if 90 < c.lon < 110)         # cluster A: uncovered
    b = next(c for c in cands if 240 < c.lon < 260)        # cluster B: covered

    assert math.isnan(a.outflow_score)
    assert not math.isnan(b.outflow_score)
    # Uncovered: confidence == connectivity_score (no penalty for missing evidence)
    assert a.confidence == pytest.approx(a.connectivity_score)
    # Covered: blended
    assert b.confidence != b.connectivity_score
    # has_outflow_evidence property reports correctly
    assert not a.has_outflow_evidence
    assert b.has_outflow_evidence
