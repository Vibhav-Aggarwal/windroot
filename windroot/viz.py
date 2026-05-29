"""Plotting helpers (optional; requires the 'plot' extra)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .pipeline import SourceResult
from .spectro import DopplerMap


def _require_mpl():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting requires: pip install 'windroot[plot]'") from exc
    return plt


def plot_sources(
    result: SourceResult,
    doppler_map: Optional[DopplerMap] = None,
    top: int = 5,
    savepath: Optional[str] = None,
    focus: bool = True,
):
    """Footpoint cloud + (optional) outflow background + ranked candidates.

    ``focus`` zooms the axes to the footpoint cloud (with padding) when it is
    compact; otherwise the full synoptic disk is shown.
    """
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    if doppler_map is not None:
        extent = [doppler_map.lon.min(), doppler_map.lon.max(),
                  doppler_map.lat.min(), doppler_map.lat.max()]
        im = ax.imshow(
            doppler_map.upflow, origin="lower", extent=extent, aspect="auto",
            cmap="RdBu_r", vmin=-np.nanmax(np.abs(doppler_map.upflow)),
            vmax=np.nanmax(np.abs(doppler_map.upflow)), alpha=0.7,
        )
        fig.colorbar(im, ax=ax, label=f"upflow [km/s] {doppler_map.line}")

    ax.scatter(result.lon_phot, result.lat_phot, s=2, c="0.4", alpha=0.25,
               label="footpoint MC cloud")

    for i, c in enumerate(result.candidates[:top], 1):
        ax.plot(c.lon, c.lat, "*", ms=16, mec="k",
                mfc=plt.cm.viridis(c.confidence))
        ax.annotate(f"#{i} ({c.confidence:.2f})", (c.lon, c.lat),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax.set_xlabel("Carrington longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_title(f"windroot source regions — {result.stream.label or 'stream'}")

    lon_span = result.lon_phot.max() - result.lon_phot.min()
    lat_span = result.lat_phot.max() - result.lat_phot.min()
    if focus and lon_span < 120 and lat_span < 80:
        padx = max(10.0, 0.4 * lon_span)
        pady = max(10.0, 0.6 * lat_span + 10.0)
        ax.set_xlim(result.lon_phot.min() - padx, result.lon_phot.max() + padx)
        ax.set_ylim(
            max(-90, result.lat_phot.min() - pady),
            min(90, result.lat_phot.max() + pady),
        )
    else:
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig, ax
