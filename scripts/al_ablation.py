"""Ablation of the six RehabAL composite-acquisition terms (U/T/A/R/C/D).

Performs a leave-one-out ablation to show which of the six RehabAL terms
actually contributes to label efficiency, especially at low budgets.  Seven
strategy variants are compared under the SAME iterative AL protocol used by
``al_iterative.py``:

  * ``rehabal_full``          - all six terms (default weights)
  * ``rehabal_no_uncertainty`` - uncertainty weight zeroed
  * ``rehabal_no_temporal``    - temporal weight zeroed
  * ``rehabal_no_angle``       - angle weight zeroed
  * ``rehabal_no_rom``         - rom weight zeroed
  * ``rehabal_no_compensation``- compensation weight zeroed
  * ``rehabal_no_rarity``      - rarity weight zeroed

Protocol (per variant, per seed):
  - Shared random init across all variants (same seed -> same initial set).
  - Round r: train on current labeled set -> record (budget, mAP); score pool
    with that model; acquire the next increment via the variant's strategy.

Outputs under ``outputs/experiments/ablation/``:
  * ``curve.csv``   - (strategy, budget, seed, n_labeled, pose_map50, pose_map50_95)
  * ``summary.json``- per-variant AUC mean+/-std + per-budget mean+/-std over seeds

Run is resumable: already-completed (variant, budget, seed) cells are skipped.

Usage::

    python scripts/al_ablation.py
    python scripts/al_ablation.py --smoke
    python scripts/al_ablation.py --help
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Make the scripts/ directory importable so al_budget_curve helpers can be
# imported (same pattern as al_iterative.py).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from al_budget_curve import (  # noqa: E402 (heavy only at runtime, not import time)
    DATASET,
    MANIFEST,
    _build_pool,
    _free_cuda,
    _recording_meta,
    _train_eval as _default_train_eval,
)
from rehabal_pose.al import DEFAULT_WEIGHTS
from rehabal_pose.al.rehabal import RehabALStrategy
from rehabal_pose.data import ManifestBuilder
from rehabal_pose.metrics.learning_curve import area_under_learning_curve
from rehabal_pose.models.yolo_pose import YoloPoseModel
from rehabal_pose.utils.io import ensure_dir, write_json
from rehabal_pose.utils.logging import get_logger
from rehabal_pose.utils.seed import set_seed

_LOG = get_logger("al_ablation")

# ---------------------------------------------------------------------------
# Variant factory
# ---------------------------------------------------------------------------

#: The six ablatable term names (canonical order matching DEFAULT_WEIGHTS keys).
ABLATION_TERMS: list[str] = [
    "uncertainty",
    "temporal",
    "angle",
    "rom",
    "compensation",
    "rarity",
]


def build_ablation_variants() -> dict[str, RehabALStrategy]:
    """Return the seven RehabAL strategy objects for the ablation study.

    The first key is ``rehabal_full`` (all default weights).  The remaining
    six are leave-one-out variants: ``rehabal_no_<term>`` where that term's
    weight is set to 0 (the remaining five are re-normalised to sum to 1).

    Returns
    -------
    dict[str, RehabALStrategy]
        Ordered dict: ``rehabal_full`` first, then the six LOO variants.
    """
    variants: dict[str, RehabALStrategy] = {}

    # Full variant — no weight override; RehabALStrategy uses DEFAULT_WEIGHTS.
    variants["rehabal_full"] = RehabALStrategy()

    # Leave-one-out variants.
    for term in ABLATION_TERMS:
        weight_override = {term: 0.0}
        variants[f"rehabal_no_{term}"] = RehabALStrategy(weights=weight_override)

    return variants


# ---------------------------------------------------------------------------
# Core scoring helper (analogous to al_iterative._score_pool)
# ---------------------------------------------------------------------------


def _score_pool_ablation(pool_imgs: list[str], weights_path: str,
                          rec_meta: dict) -> object:
    """Load model from *weights_path*, score the pool, return a Pool."""
    model = YoloPoseModel().load(weights_path)
    pool = _build_pool(pool_imgs, model, rec_meta)
    del model
    _free_cuda()
    return pool


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None, *, train_eval: Optional[Callable] = None) -> int:
    """Entry point for the ablation runner.

    Parameters
    ----------
    argv:
        Argument list (``None`` → ``sys.argv[1:]``).
    train_eval:
        Injectable training/evaluation hook.  Signature must match
        ``al_budget_curve._train_eval(selected_imgs, out_dir, epochs, imgsz,
        seed, batch) -> dict``.  Defaults to the real YOLO trainer.
        Injecting a fake here is how the test suite avoids GPU usage.
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--budgets", nargs="+", type=float,
        default=[0.05, 0.10, 0.20, 0.40],
        help="label-budget fractions to evaluate (default: 0.05 0.10 0.20 0.40)",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2],
        help="random seeds (default: 0 1 2)",
    )
    ap.add_argument(
        "--epochs", type=int, default=10,
        help="training epochs per round (default: 10)",
    )
    ap.add_argument(
        "--imgsz", type=int, default=416,
        help="input image size (default: 416)",
    )
    ap.add_argument(
        "--batch", type=int, default=16,
        help="training batch size (default: 16)",
    )
    ap.add_argument(
        "--out", default="outputs/experiments/ablation",
        help="output directory (default: outputs/experiments/ablation)",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help=(
            "smoke-test mode: 2 variants (rehabal_full + rehabal_no_uncertainty), "
            "2 budgets (0.05, 0.10), 1 epoch, 1 seed"
        ),
    )
    args = ap.parse_args(argv)

    if args.smoke:
        # Keep only two variants, two budgets, one epoch, one seed.
        args.seeds = [0]
        args.budgets = [0.05, 0.10]
        args.epochs = 1

    _train_eval = train_eval if train_eval is not None else _default_train_eval

    out_root = ensure_dir(Path(args.out))
    curve_csv = out_root / "curve.csv"

    # --- pool images ---------------------------------------------------------
    pool_imgs = sorted(glob.glob(str(DATASET / "images" / "train" / "*.jpg")))
    if not pool_imgs:
        _LOG.error("No training images under %s; build the dataset first.", DATASET)
        return 1
    N = len(pool_imgs)
    budgets = sorted(args.budgets)
    targets = [max(1, int(round(b * N))) for b in budgets]
    _LOG.info("Pool %d imgs; budgets %s -> targets %s", N, budgets, targets)

    # --- manifest / metadata -------------------------------------------------
    rec_meta = _recording_meta(ManifestBuilder.load(MANIFEST))

    # --- build strategy variants ---------------------------------------------
    variants = build_ablation_variants()
    strategy_names = list(variants.keys())
    if args.smoke:
        strategy_names = ["rehabal_full", "rehabal_no_uncertainty"]

    _LOG.info("Ablation variants: %s", strategy_names)

    # --- load already-completed rows for resumability ------------------------
    if curve_csv.exists():
        existing_df = pd.read_csv(curve_csv)
        done_cells: set[tuple] = set(
            zip(existing_df["strategy"], existing_df["budget"].astype(float),
                existing_df["seed"].astype(int))
        )
        rows = existing_df.to_dict("records")
        _LOG.info("Resuming: %d existing rows, %d done cells", len(rows), len(done_cells))
    else:
        existing_df = pd.DataFrame()
        done_cells = set()
        rows = []

    # --- main loop -----------------------------------------------------------
    for seed in args.seeds:
        # Shared random init (same across all variants for a fair comparison).
        init_rng = np.random.default_rng(1000 + seed)
        init_idx = list(init_rng.choice(N, size=targets[0], replace=False))

        for strategy_name in strategy_names:
            strat = variants[strategy_name]

            # Reset labeled set to the shared init for this seed.
            labeled = list(init_idx)
            model_weights: Optional[str] = None  # previous round's trained weights

            for r, (budget, target) in enumerate(zip(budgets, targets)):
                cell = (strategy_name, float(budget), int(seed))
                tag = f"{strategy_name}_b{budget:g}_s{seed}"
                cell_dir = ensure_dir(out_root / tag)
                idx_json = cell_dir / "labeled_idx.json"

                # Skip if already computed (resumable).
                if cell in done_cells:
                    # Restore the exact labeled-index list so the next round's
                    # acquisition starts from the correct prefix.
                    if idx_json.exists():
                        with idx_json.open(encoding="utf-8") as fh:
                            labeled = json.load(fh)
                        _LOG.info("[%s b=%g s=%d] resumed labeled set (%d imgs) "
                                  "from labeled_idx.json",
                                  strategy_name, budget, seed, len(labeled))
                    else:
                        _LOG.warning("[%s b=%g s=%d] done but labeled_idx.json "
                                     "missing — labeled set stays at previous round",
                                     strategy_name, budget, seed)
                    continue

                # Acquire up to target using the PREVIOUS round's model.
                if len(labeled) < target:
                    if model_weights is None:
                        # Cold start: random fill (no model yet).
                        fill_rng = np.random.default_rng(seed * 97 + r)
                        rem = [i for i in range(N) if i not in set(labeled)]
                        labeled += list(
                            fill_rng.choice(rem,
                                            size=min(target - len(labeled), len(rem)),
                                            replace=False)
                        )
                    else:
                        pool = _score_pool_ablation(pool_imgs, model_weights, rec_meta)
                        cand = [i for i in range(N) if i not in set(labeled)]
                        set_seed(seed * 97 + r)
                        sel = strat.select(pool, target - len(labeled), candidates=cand)
                        labeled += [i for i in sel if i not in set(labeled)]
                        # Top up if the strategy returned fewer than requested.
                        if len(labeled) < target:
                            fill_rng = np.random.default_rng(seed * 13 + r)
                            rem = [i for i in range(N) if i not in set(labeled)]
                            labeled += list(
                                fill_rng.choice(
                                    rem,
                                    size=min(target - len(labeled), len(rem)),
                                    replace=False,
                                )
                            )

                sel_imgs = [pool_imgs[i] for i in labeled]
                _LOG.info("[%s] round %d: train on %d imgs", tag, r, len(sel_imgs))

                res = _train_eval(sel_imgs, cell_dir, args.epochs,
                                  args.imgsz, seed, args.batch)
                model_weights = res.get("weights")

                # Persist the exact labeled-index list for correct resume.
                # Cast to plain int so json.dump handles numpy int64 scalars.
                with idx_json.open("w", encoding="utf-8") as fh:
                    json.dump(sorted(int(i) for i in labeled), fh)

                rows.append({
                    "strategy": strategy_name,
                    "budget": budget,
                    "seed": seed,
                    "n_labeled": len(sel_imgs),
                    "pose_map50": res.get("map50"),
                    "pose_map50_95": res.get("map50_95"),
                })
                done_cells.add(cell)

                # Persist incrementally.
                pd.DataFrame(rows).to_csv(curve_csv, index=False)

                _nn = lambda v: float("nan") if v is None else float(v)
                _LOG.info("[%s] pose mAP50=%.4f  mAP50-95=%.4f", tag,
                          _nn(res.get("map50")), _nn(res.get("map50_95")))

    # --- summary -------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(curve_csv, index=False)

    summary: dict = {}
    for strategy_name in strategy_names:
        sub_df = df[df["strategy"] == strategy_name]

        # Per-budget mean ± std across seeds.
        budget_stats: dict = {}
        for b in budgets:
            pts = sub_df[sub_df["budget"].astype(float) == float(b)][
                "pose_map50_95"
            ].dropna().values
            budget_stats[str(b)] = {
                "mean": float(np.mean(pts)) if len(pts) else float("nan"),
                "std": float(np.std(pts, ddof=0)) if len(pts) else float("nan"),
                "n": int(len(pts)),
            }

        # AUC over budget, averaged per seed first then mean ± std.
        # NaN rows are dropped (consistent with run_paper_experiments policy).
        auc_per_seed: list[float] = []
        for seed in args.seeds:
            seed_df = (
                sub_df[sub_df["seed"] == seed]
                .dropna(subset=["pose_map50_95"])
                .sort_values("budget")
            )
            if len(seed_df) >= 2:
                auc_val = area_under_learning_curve(
                    seed_df["budget"].tolist(),
                    seed_df["pose_map50_95"].tolist(),
                    normalize=True,
                )
                auc_per_seed.append(float(auc_val))

        auc_mean = float(np.mean(auc_per_seed)) if auc_per_seed else float("nan")
        auc_std = float(np.std(auc_per_seed, ddof=0)) if len(auc_per_seed) > 1 else 0.0

        summary[strategy_name] = {
            "auc_map50_95_mean": auc_mean,
            "auc_map50_95_std": auc_std,
            "auc_per_seed": auc_per_seed,
            "budget_stats": budget_stats,
        }

    write_json(out_root / "summary.json", summary)
    _LOG.info(
        "Ablation AUC (mean±std, mAP50-95): %s",
        {s: f"{v['auc_map50_95_mean']:.4f}±{v['auc_map50_95_std']:.4f}"
         for s, v in summary.items()},
    )

    print("AL_ABLATION_DONE", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
