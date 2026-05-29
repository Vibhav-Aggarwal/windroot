"""Command-line interface: ``windroot find ...``."""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .insitu import InSituStream
from .pipeline import find_sources


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="windroot",
        description="Find solar-wind source regions by fusing magnetic connectivity "
        "with spectroscopic outflow evidence.",
    )
    p.add_argument("--version", action="version", version=f"windroot {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("find", help="back-map an in-situ stream to candidate source regions")
    f.add_argument("--vsw", type=float, required=True, help="radial solar-wind speed [km/s]")
    f.add_argument("--lon", type=float, required=True, help="spacecraft Carrington longitude [deg]")
    f.add_argument("--lat", type=float, default=0.0, help="spacecraft latitude [deg] (default 0)")
    f.add_argument("--r-au", type=float, default=1.0, help="heliocentric distance [AU] (default 1)")
    f.add_argument("--vsw-err", type=float, default=30.0, help="1-sigma speed error [km/s]")
    f.add_argument("--r-ss", type=float, nargs=2, default=(1.5, 3.5),
                   metavar=("MIN", "MAX"), help="source-surface height range [Rsun]")
    f.add_argument("--n-mc", type=int, default=4000, help="Monte-Carlo samples")
    f.add_argument("--top", type=int, default=5, help="how many candidates to print")
    f.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    return p


def main(argv=None) -> int:
    import numpy as np

    args = _build_parser().parse_args(argv)
    if args.command == "find":
        stream = InSituStream.from_au(
            v_sw=args.vsw, r_au=args.r_au, lon_obs=args.lon, lat_obs=args.lat,
            v_sw_err=args.vsw_err, label=f"v={args.vsw:.0f} lon={args.lon:.0f}",
        )
        rng = np.random.default_rng(args.seed) if args.seed is not None else None
        result = find_sources(stream, n_mc=args.n_mc, r_ss_range=tuple(args.r_ss), rng=rng)
        print(f"# windroot {__version__} — source regions (connectivity-only; "
              f"add a Doppler map for the spectroscopic layer)")
        print(f"# stream: {stream.label}, r={args.r_au} AU")
        ens = result.ensemble.summary()
        print(f"# source-surface footpoint: lon={ens['lon_ss_mean']:.1f}"
              f"+/-{ens['lon_ss_std']:.1f} deg, lat={ens['lat_ss_mean']:.1f} deg")
        print(result.to_table() if result.candidates else "  (no candidates)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
