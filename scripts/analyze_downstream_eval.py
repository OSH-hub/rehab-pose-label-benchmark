"""Usage: python scripts/analyze_downstream_eval.py"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from rehabal_pose.utils.logging import get_logger  # noqa: E402

_LOG = get_logger("downstream_analysis")

COND_LABEL = {
    "A": "A: pseudo-label after shared GT init",
    "B": "B: AL + simulated correction + segmentation-based reject",
    "ref": "ref: AL + full-GT comparator",
}
COND_ORDER = ["A", "B", "ref"]


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def _parse_name(stem: str) -> tuple[str, int] | None:
    name = stem.replace("_downstream", "")
    if "_s" not in name:
        return None
    cond, _, seed = name.rpartition("_s")
    try:
        return cond, int(seed)
    except ValueError:
        return None


def _mean_std(vals) -> tuple[float, float]:
    arr = np.array([v for v in vals if v == v], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="outputs/experiments/downstream")
    ap.add_argument("--out-md", default="verification/downstream_results.md")
    ap.add_argument("--out-png", default="outputs/experiments/downstream/downstream_angle_mae.png")
    args = ap.parse_args(argv)

    ddir = Path(args.dir)
    files = sorted(ddir.glob("*_downstream.json"))
    if not files:
        _LOG.error("No *_downstream.json under %s - run run_downstream_eval first.", ddir)
        return 1

    METRICS = ["joint_angle_mae_mean", "rom_error_mean", "smoothness_error_mean"]
    data: dict = {m: defaultdict(lambda: defaultdict(list)) for m in METRICS}
    seeds_seen: set = set()
    budgets_seen: set = set()
    failures: list[str] = []
    for f in files:
        parsed = _parse_name(f.stem)
        if parsed is None:
            _LOG.warning("skip unparseable %s", f.name)
            continue
        cond, seed = parsed
        seeds_seen.add(seed)
        d = json.loads(f.read_text(encoding="utf-8"))
        for _bk, bv in d.get("budgets", {}).items():
            if "error" in bv:
                failures.append(f"{f.name}:{_bk}: {bv['error']}")
                continue
            b = float(bv["budget"])
            budgets_seen.add(b)
            agg = bv.get("aggregate", {})
            for m in METRICS:
                val = agg.get(m)
                if isinstance(val, dict):
                    val = val.get("mean")
                if val is not None:
                    data[m][cond][b].append(float(val))

    if failures:
        for failure in failures:
            _LOG.error("Invalid input result: %s", failure)
        return 1

    conds = [c for c in COND_ORDER if any(c in data[m] for m in METRICS)]
    budgets = sorted(budgets_seen)
    seeds = sorted(seeds_seen)
    _LOG.info("conditions=%s budgets=%s seeds=%s", conds, budgets, seeds)

    METRIC_TITLE = {
        "joint_angle_mae_mean": "Joint-angle MAE (deg, lower=better)",
        "rom_error_mean": "ROM error (deg, lower=better)",
        "smoothness_error_mean": (
            "Mean jerk-magnitude error (image-diagonal units s^-3, lower=better)"
        ),
    }

    lines = ["# Downstream Kinematic-Fidelity Results",
             "",
             f"Per-budget downstream metrics on the fixed REHAB24-6 val set "
             f"(seeds {seeds}). Predicted and GT keypoints are mapped to an "
             f"image-diagonal-normalized frame before scoring; recording frames "
             f"are ordered by numeric source-frame index at 20/30-s intervals. "
             f"Only spatial (x,y) columns enter joint angles. Lower is better "
             f"for every metric.",
             ""]

    for m in METRICS:
        lines += [f"## {METRIC_TITLE[m]} (mean ± std over seeds)", "",
                  "| budget | " + " | ".join(conds) + " |",
                  "|---|" + "|".join(["---"] * len(conds)) + "|"]
        for b in budgets:
            row = [f"{b:g}"]
            for c in conds:
                mu, sd = _mean_std(data[m][c].get(b, []))
                row.append(f"{mu:.3f} ± {sd:.3f}" if mu == mu else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    m = "joint_angle_mae_mean"
    if budgets and {"A", "B"} <= set(conds):
        b = budgets[-1]
        a_mu, _ = _mean_std(data[m]["A"].get(b, []))
        b_mu, _ = _mean_std(data[m]["B"].get(b, []))
        lines += ["## Primary downstream comparison", "",
                  f"- At budget {b:g}, joint-angle MAE: A={a_mu:.2f}deg, "
                  f"B={b_mu:.2f}deg; B-A={b_mu - a_mu:+.2f}deg "
                  f"(negative values favor B).",
                  "- These downstream metrics quantify kinematic fidelity on "
                  "the held-out benchmark; they are not clinical outcome "
                  "measures.",
                  ""]

    md = "\n".join(lines) + "\n"
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    _LOG.info("Wrote %s", out_md)
    _safe_print(md)

    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from _figstyle import apply_pub_style, style_for, save_fig
        apply_pub_style(plt)
        plt.figure(figsize=(7, 5))
        for c in conds:
            ys, es = [], []
            for b in budgets:
                mu, sd = _mean_std(data[m][c].get(b, []))
                ys.append(mu); es.append(sd if sd == sd else 0.0)
            ys = np.array(ys); es = np.array(es)
            st = style_for(c)
            plt.plot(budgets, ys, linewidth=2.5 if c == "B" else 1.8,
                     label=COND_LABEL.get(c, c), **st)
            plt.fill_between(budgets, ys - es, ys + es, alpha=0.15, color=st.get("color"))
        plt.xlabel("Label budget (fraction of pool)")
        plt.ylabel("Joint-angle MAE (°)")
        plt.title("Downstream kinematic fidelity: joint-angle error vs label budget")
        plt.legend(loc="upper right", framealpha=1.0)
        plt.tight_layout()
        out_png = save_fig(plt, args.out_png)
        _LOG.info("Wrote %s", out_png)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("plot skipped: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
