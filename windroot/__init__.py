"""windroot — observation-driven solar-wind source-region finder.

Fuses magnetic connectivity (PFSS + ballistic back-mapping) with spectroscopic
outflow evidence (SPICE/EIS Doppler maps) to identify and rank where on the Sun a
given solar-wind stream was born.

Quick start
-----------
>>> from windroot import InSituStream, find_sources
>>> stream = InSituStream.near_earth(v_sw=420, lon_obs=120, lat_obs=3, v_sw_err=30)
>>> result = find_sources(stream)
>>> print(result.to_table())
"""
from ._units import DEFAULT_R_SS, SIDEREAL_ROTATION_PERIOD
from .ballistic import (
    BallisticEnsemble,
    BallisticResult,
    ballistic_backmap,
    ballistic_backmap_mc,
    corotation_shift,
)
from .insitu import InSituStream
from .pfss import DipoleSourceSurface, Footpoint, map_footpoints, pfss_connectivity
from .pipeline import SourceResult, find_sources
from .rank import SourceCandidate, rank_sources
from .spectro import DopplerMap

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_R_SS",
    "SIDEREAL_ROTATION_PERIOD",
    "InSituStream",
    "find_sources",
    "SourceResult",
    "ballistic_backmap",
    "ballistic_backmap_mc",
    "corotation_shift",
    "BallisticResult",
    "BallisticEnsemble",
    "DipoleSourceSurface",
    "Footpoint",
    "map_footpoints",
    "pfss_connectivity",
    "DopplerMap",
    "SourceCandidate",
    "rank_sources",
]
