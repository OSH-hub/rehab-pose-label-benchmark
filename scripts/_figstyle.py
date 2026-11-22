"""Shared publication figure style for the analyze_*.py manuscript plots.

matplotlib-only, no heavy imports at module load (receives ``plt`` as an
argument), so it is safe to import from inside the guarded plot blocks.

Design goals (from the SCI figure audit):
  * one condition->style mapping used by EVERY figure, so A/B/ref never swap
    colors between figures;
  * colorblind-safe (Okabe-Ito palette) AND grayscale-distinguishable
    (distinct linestyle + marker per condition);
  * print-ready output (600 dpi raster + vector PDF).
"""
from __future__ import annotations

from pathlib import Path

# Okabe-Ito colorblind-safe palette + distinct linestyle/marker per condition
# so the figures also read correctly in grayscale print.
COND_STYLE = {
    "A":   dict(color="#0072B2", linestyle="--", marker="s"),  # blue
    "B":   dict(color="#D55E00", linestyle="-",  marker="o"),  # vermillion
    "ref": dict(color="#009E73", linestyle=":",  marker="^"),  # bluish green
}

_RC = {
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "axes.grid": True, "grid.alpha": 0.3,
}

_METRIC_LABEL = {
    "map50_95": "mAP@[.50:.95]",
    "map50": "mAP@50",
}


def apply_pub_style(plt) -> None:
    """Apply the shared rcParams to a pyplot module."""
    plt.rcParams.update(_RC)


def style_for(cond: str) -> dict:
    """Per-condition plot kwargs (color/linestyle/marker). Copy, safe to mutate."""
    return dict(COND_STYLE.get(cond, {}))


def metric_label(metric: str) -> str:
    """Render a code-style metric key (e.g. ``map50_95``) as a publication label."""
    return _METRIC_LABEL.get(str(metric), str(metric))


def save_fig(plt, out_png) -> Path:
    """Save a print-ready raster (600 dpi) plus a vector PDF alongside it."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    return out_png
