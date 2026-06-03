"""Run the full windroot backtest suite and write a summary report."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPORT_PATH = Path(__file__).resolve().parent.parent / "BACKTEST_REPORT.md"


def _capture(callable_, *args, **kwargs) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            rc = callable_(*args, **kwargs)
        except SystemExit as e:
            rc = e.code or 0
    return int(rc or 0), buf.getvalue()


def main() -> int:
    from backtests.run_synthetic import main as run_a
    rc_a, out_a = _capture(run_a)
    print(f"== A. Synthetic multi-case  -> rc={rc_a} ==")
    print(out_a)

    have_pfss = True
    try:
        from sunkit_magex.pfss import Input  # noqa: F401
    except ImportError:
        have_pfss = False

    if have_pfss:
        from backtests.run_pfss_xv import main as run_b
        rc_b, out_b = _capture(run_b)
        print(f"\n== B. Dipole-vs-sunkit-magex cross-validation -> rc={rc_b} ==")
        print(out_b)
        from backtests.run_real_gong import main as run_c
        rc_c, out_c = _capture(run_c)
        print(f"\n== C. Real GONG synoptic magnetogram (CR 2234) -> rc={rc_c} ==")
        print(out_c)
    else:
        rc_b, out_b = -1, "(skipped: sunkit-magex not installed)"
        rc_c, out_c = -1, "(skipped: sunkit-magex not installed)"

    report = (
        "# windroot backtest report\n\n"
        "## A. Synthetic multi-case (analytic-dipole known truth)\n\n"
        "```\n" + out_a + "\n```\n\n"
        "## B. Cross-validation: analytic dipole vs sunkit-magex PFSS\n\n"
        "```\n" + out_b + "\n```\n\n"
        "## C. End-to-end on a real GONG synoptic magnetogram (CR 2234)\n\n"
        "```\n" + out_c + "\n```\n\n"
        f"## Summary\n\n"
        f"- A return code: `{rc_a}` (0 = pass)\n"
        f"- B return code: `{rc_b}`\n"
        f"- C return code: `{rc_c}`\n"
    )
    REPORT_PATH.write_text(report)
    print(f"\nReport written: {REPORT_PATH}")
    return 0 if (rc_a == 0 and rc_b in (0, -1) and rc_c in (0, -1)) else 1


if __name__ == "__main__":
    sys.exit(main())
