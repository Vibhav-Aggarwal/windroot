import numpy as np
import pytest

from windroot.cli import main


def test_cli_find_runs_and_prints(capsys):
    rc = main(["find", "--vsw", "420", "--lon", "120", "--lat", "3", "--seed", "1", "--top", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "source regions" in out
    assert "source-surface footpoint" in out
    assert "rank" in out


def test_cli_version():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_viz_smoke(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    from windroot import DopplerMap, InSituStream, find_sources
    from windroot.viz import plot_sources

    stream = InSituStream.near_earth(v_sw=400, lon_obs=120, lat_obs=2, v_sw_err=40)
    glon = np.arange(0, 360, 4.0)
    glat = np.arange(-60, 61, 4.0)
    LON, LAT = np.meshgrid(glon, glat)
    dm = DopplerMap(glon, glat, 20.0 * np.exp(-(((LON - 180) / 8) ** 2 + (LAT / 8) ** 2)))
    result = find_sources(stream, doppler_map=dm, n_mc=1500, rng=np.random.default_rng(3))

    out = tmp_path / "sources.png"
    fig, ax = plot_sources(result, doppler_map=dm, savepath=str(out))
    assert out.exists()
    assert ax.get_xlabel().lower().startswith("carrington")
