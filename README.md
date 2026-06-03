# windroot

**Observation-driven solar-wind source-region finder.**

For a gust of solar wind measured out in space, *where on the Sun was it born?*
`windroot` answers this by **fusing two independent lines of evidence**:

1. **Magnetic connectivity** — PFSS field extrapolation + ballistic back-mapping
   (where the magnetic field says the wind came from), and
2. **Spectroscopic outflow** — SPICE / Hinode-EIS Doppler maps (where plasma is
   *actually observed* flowing outward, the candidate seed of the wind).

## Why this exists

Source-region mapping is today done with **magnetic geometry alone** — PFSS +
ballistic back-mapping (Badman, Dakeyo, Koukras; tools `sunkit-magex`, `solar-mach`).
The spectroscopic evidence — EUV Doppler maps that physically show outflowing
plasma — exists only as separate papers and is **never fused into the ranking**.
`windroot` adds that missing evidence layer: it ranks candidate source regions by
how well magnetic connectivity *and* observed outflow agree. No released package
does this.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # core (numpy, scipy, astropy) — works everywhere
pip install -e ".[all]"       # + PFSS/data/plot/ml integrations
```

The **core runs with no heavy dependencies**; real-magnetogram PFSS, data fetching,
plotting and ML are optional extras (`pfss`, `data`, `insitu`, `spice`, `eis`,
`connect`, `plot`, `ml`).

## Quick start

```python
from windroot import InSituStream, find_sources

# a slow stream observed near Earth
stream = InSituStream.near_earth(v_sw=360, lon_obs=140, lat_obs=4, v_sw_err=35)

result = find_sources(stream, n_mc=6000)   # connectivity only
print(result.to_table())
```

Add the spectroscopic layer:

```python
from windroot import DopplerMap
# upflow: 2-D (lat, lon) outflow speed [km/s], positive = outward (blueshift)
dm = DopplerMap.from_vlos(lon, lat, vlos, line="Ne VIII 770")
fused = find_sources(stream, doppler_map=dm, w_connectivity=0.4, w_outflow=0.6)
print(fused.best)            # highest-confidence source region
```

CLI:

```bash
windroot find --vsw 360 --lon 140 --lat 4 --vsw-err 35 --top 5
```

Worked example (runs on the core alone, saves a figure):

```bash
python examples/worked_example_AR_to_PSP.py
```

## Pipeline

```
in-situ stream ──▶ ballistic back-map (MC over v_sw and R_ss)
                       │
                       ▼
              source-surface footpoints ──▶ PFSS map-down ──▶ photospheric cloud
                                                                   │
                  SPICE/EIS Doppler outflow map ───────────────────┤
                                                                   ▼
                                              fuse + rank ──▶ ranked source regions
                                                              (connectivity × outflow,
                                                               with confidence)
```

| Module | Role |
|--------|------|
| `insitu.py` | in-situ stream record (OMNI/PSP/Wind) |
| `ballistic.py` | ballistic back-map to the source surface + Monte-Carlo uncertainty (dominant systematic: source-surface height) |
| `pfss.py` | analytic dipole source-surface (closed-form footpoint mapping) + numeric tracer; optional `sunkit-magex` backend |
| `spectro.py` | `DopplerMap` outflow evidence; optional SPICE skew-correction hook (Shrivastav et al. 2026, A&A 706) |
| `rank.py` | fuse connectivity × outflow → ranked `SourceCandidate`s |
| `pipeline.py` | `find_sources()` end-to-end |
| `viz.py`, `cli.py` | plotting / command line |

## Validation

```bash
pip install -e ".[dev]" && pytest --cov=windroot
```

33 tests, ~95% coverage. The key physics checks:
- the **numeric field-line tracer reproduces the closed-form dipole footpoint mapping**
  (synthetic known-source recovery) and conserves the field-line invariant;
- ballistic corotation matches a hand calculation; faster wind → smaller shift;
- the **spectroscopic layer re-ranks** a magnetically-weaker candidate above a
  magnetically-stronger one when outflow is observed there (the core claim).

**Multi-dataset backtest suite** (see `backtests/` and [BACKTEST_REPORT.md](BACKTEST_REPORT.md)):

| | what | result |
|---|---|---|
| **A** | synthetic multi-case: recovery of analytic dipole truth across varying speed/latitude, robustness to a 60-deg spurious outflow decoy, and spectroscopic discrimination under wide-MC ambiguity | **27/27 pass** (9 + 9 + 9) |
| **B** | cross-validate windroot's analytic dipole footpoint mapping against an independent numerical PFSS (sunkit-magex) on a synthetic dipolar magnetogram, 64 seeds | **RMS dlon = 0.00 deg, dlat = 0.23 deg** |
| **C** | full pipeline on a real GONG synoptic magnetogram (CR 2234, 2020-09-01) for 6 streams across Carrington longitudes | **6/6 streams produce PFSS candidates; median 16.1 deg separation from the dipole result** (PFSS is using real structure) |

Run the suite (needs the `pfss` extra):
```bash
python -m backtests.run_all       # writes BACKTEST_REPORT.md
```

**Next step (real-data PSP/OMNI events):** reproduce a published PSP source-region
conjunction (Badman/Dakeyo) with a CR-matched GONG/ADAPT magnetogram, then
overlay a real SPICE/EIS Doppler map. Backtest C is the scaffolding; supplying
the published event metadata is the only remaining work.

## Known limitations

- The built-in **analytic dipole** field is axisymmetric, so footpoint *latitude*
  is degenerate (candidates differ in longitude only). Real, non-axisymmetric
  source regions require the `sunkit-magex` magnetogram backend.
- Ballistic back-mapping assumes constant radial speed; source-surface height is
  the dominant uncertainty and is Monte-Carloed, not solved.
- The outflow→source weighting is empirical; the optional `ml.py` layer (WSA+-style)
  is the planned data-driven successor.

## Authors

Vibhav Aggarwal, Arpit Kumar Shrivastav (intended co-author).

## License

BSD-3-Clause.
