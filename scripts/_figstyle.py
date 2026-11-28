from __future__ import annotations

from pathlib import Path

COND_STYLE = {
    "A":   dict(color="#0072B2", linestyle="--", marker="s"),
    "B":   dict(color="#D55E00", linestyle="-",  marker="o"),
    "ref": dict(color="#009E73", linestyle=":",  marker="^"),
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
    plt.rcParams.update(_RC)


def style_for(cond: str) -> dict:
    return dict(COND_STYLE.get(cond, {}))


def metric_label(metric: str) -> str:
    return _METRIC_LABEL.get(str(metric), str(metric))


def save_fig(plt, out_png) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    return out_png
