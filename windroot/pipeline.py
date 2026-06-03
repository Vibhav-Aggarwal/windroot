"""End-to-end orchestration: in-situ stream -> source-region candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._units import DEFAULT_R_SS
from .ballistic import BallisticEnsemble, ballistic_backmap_mc
from .insitu import InSituStream
from .pfss import DipoleSourceSurface, map_footpoints
from .rank import SourceCandidate, rank_sources
from .spectro import DopplerMap


@dataclass
class SourceResult:
    """Full result of a source-region search."""

    candidates: list[SourceCandidate]
    ensemble: BallisticEnsemble
    lon_phot: np.ndarray
    lat_phot: np.ndarray
    stream: InSituStream

    @property
    def best(self) -> Optional[SourceCandidate]:
        return self.candidates[0] if self.candidates else None

    def to_table(self) -> str:
        lines = [f"{'rank':>4} {'lon':>7} {'lat':>7} {'conn':>6} {'outflow':>8} {'conf':>6}"]
        for i, c in enumerate(self.candidates, 1):
            outflow = f"{c.outflow_kms:7.1f}" if np.isfinite(c.outflow_kms) else "    n/a"
            lines.append(
                f"{i:>4} {c.lon:7.1f} {c.lat:7.1f} {c.connectivity_score:6.2f} "
                f"{outflow} {c.confidence:6.2f}"
            )
        return "\n".join(lines)


def find_sources(
    stream: InSituStream,
    field: Optional[DipoleSourceSurface] = None,
    doppler_map: Optional[DopplerMap] = None,
    n_mc: int = 2000,
    r_ss_range=(1.5, 3.5),
    bin_deg: float = 5.0,
    sample_radius_deg: float = 5.0,
    w_connectivity: float = 0.5,
    w_outflow: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> SourceResult:
    """Find and rank solar-wind source regions for an in-situ stream.

    Steps: ballistic back-map (Monte-Carlo over speed + source-surface height) ->
    PFSS footpoint mapping -> fuse connectivity with spectroscopic outflow -> rank.

    With ``field=None`` an analytic dipole source surface is used, so the pipeline
    runs end-to-end with zero external data (useful for demos and tests). Pass a
    sunkit-magex-backed field for real magnetograms.
    """
    field = field or DipoleSourceSurface(r_ss=float(np.mean(r_ss_range)))

    ensemble = ballistic_backmap_mc(
        v_sw=stream.v_sw,
        r_obs=stream.r_obs,
        lon_obs=stream.lon_obs,
        lat_obs=stream.lat_obs,
        v_sw_err=stream.v_sw_err,
        r_ss_range=r_ss_range,
        n=n_mc,
        rng=rng,
    )

    if hasattr(field, "map_footpoints"):
        lon_phot, lat_phot = field.map_footpoints(ensemble.lon_ss, ensemble.lat_ss)
    else:
        lon_phot, lat_phot = map_footpoints(field, ensemble.lon_ss, ensemble.lat_ss)

    candidates = rank_sources(
        lon_phot,
        lat_phot,
        doppler_map=doppler_map,
        bin_deg=bin_deg,
        sample_radius_deg=sample_radius_deg,
        w_connectivity=w_connectivity,
        w_outflow=w_outflow,
    )

    return SourceResult(
        candidates=candidates,
        ensemble=ensemble,
        lon_phot=lon_phot,
        lat_phot=lat_phot,
        stream=stream,
    )
