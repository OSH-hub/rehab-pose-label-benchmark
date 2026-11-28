"""Usage: python scripts/plot_method_diagram.py"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

_DEFAULT_OUT = (
    _HERE.parent / "outputs" / "experiments" / "paper_figures" / "fig_method_overview.png"
)


def _blend(hex_color: str, amount: float) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(hex_color)
    return (
        r + (1.0 - r) * amount,
        g + (1.0 - g) * amount,
        b + (1.0 - b) * amount,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _HALO = [pe.withStroke(linewidth=2.6, foreground="white")]

    from _figstyle import COND_STYLE, apply_pub_style, save_fig

    apply_pub_style(plt)

    fig, ax = plt.subplots(figsize=(16.5, 9.0))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 9.5)
    ax.set_axis_off()

    NEUTRAL_EDGE = "#333333"
    NEUTRAL_FILL = "#F2F2F2"
    ARROW_GREY = "#444444"

    def box(cx, cy, w, h, *, edge, fill, hatch=None, lw=1.6):
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=lw,
            edgecolor=edge,
            facecolor=fill,
            hatch=hatch,
            mutation_aspect=1.0,
            zorder=3,
        )
        ax.add_patch(patch)
        return cx, cy

    def title_body(cx, cy, h, title, body, *, tcolor, title_fs=11, body_fs=8.4):
        ax.text(
            cx,
            cy + h / 2 - 0.30,
            title,
            ha="center",
            va="top",
            fontsize=title_fs,
            fontweight="bold",
            color=tcolor,
            zorder=4,
            path_effects=_HALO,
        )
        ax.text(
            cx,
            cy + h / 2 - 0.72,
            body,
            ha="center",
            va="top",
            fontsize=body_fs,
            color="#111111",
            zorder=4,
            linespacing=1.35,
            path_effects=_HALO,
        )

    def centered(cx, cy, title, body, *, title_fs=11, body_fs=8.4, tcolor="#111111"):
        ax.text(
            cx,
            cy + 0.44,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
            color=tcolor,
            zorder=4,
        )
        ax.text(
            cx,
            cy - 0.22,
            body,
            ha="center",
            va="center",
            fontsize=body_fs,
            color="#111111",
            zorder=4,
            linespacing=1.35,
        )

    def arrow(x0, y0, x1, y1, *, color=ARROW_GREY, lw=1.8, ls="-", rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=lw,
                linestyle=ls,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                zorder=2,
                shrinkA=0,
                shrinkB=0,
            )
        )

    W = 2.6
    H = 1.7
    cx_pool = 1.55
    cx_sel = 4.80
    cx_lab = 8.50
    cx_train = 11.95
    cx_eval = 14.90
    y_mid = 4.30

    lab_w = 3.0
    lab_h = 1.55
    y_A = 6.85
    y_B = 4.30
    y_ref = 1.75

    box(cx_pool, y_mid, W, H, edge=NEUTRAL_EDGE, fill=NEUTRAL_FILL)
    centered(
        cx_pool,
        y_mid,
        "REHAB24-6",
        "unlabeled pool\n8367 frames",
    )

    box(cx_sel, y_mid, W, H, edge=NEUTRAL_EDGE, fill=NEUTRAL_FILL)
    centered(
        cx_sel,
        y_mid,
        "Same acquisition rule",
        "temporal scoring\nsame budget / round",
    )

    grp_x0, grp_x1 = cx_lab - lab_w / 2 - 0.35, cx_lab + lab_w / 2 + 0.35
    grp_y0, grp_y1 = y_ref - lab_h / 2 - 0.30, y_A + lab_h / 2 + 0.45
    ax.add_patch(
        FancyBboxPatch(
            (grp_x0, grp_y0),
            grp_x1 - grp_x0,
            grp_y1 - grp_y0,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.3,
            edgecolor="#888888",
            facecolor="none",
            linestyle=(0, (5, 4)),
            zorder=1,
        )
    )
    ax.text(
        cx_lab,
        grp_y1 - 0.26,
        "Three feedback conditions; later selections may diverge",
        ha="center",
        va="center",
        fontsize=9.5,
        style="italic",
        color="#555555",
        zorder=4,
    )

    conds = [
        (
            "A",
            y_A,
            "Condition A",
            "pseudo-label self-training\nafter shared 5%\nGT initialization",
            "///",
        ),
        (
            "B",
            y_B,
            "Condition B",
            "simulated correction oracle\nGT + per-axis Gaussian noise\nreject: outside rep · extra person",
            "xxx",
        ),
        (
            "ref",
            y_ref,
            "Condition ref",
            "full-ground-truth\ncomparator",
            "\\\\\\",
        ),
    ]
    lab_centers = {}
    for key, cy, title, body, hatch in conds:
        color = COND_STYLE[key]["color"]
        box(
            cx_lab,
            cy,
            lab_w,
            lab_h,
            edge=color,
            fill=_blend(color, 0.82),
            hatch=hatch,
            lw=2.0,
        )
        title_body(cx_lab, cy, lab_h, title, body, tcolor=color)
        lab_centers[key] = cy

    box(cx_train, y_mid, W, H + 0.15, edge=NEUTRAL_EDGE, fill=NEUTRAL_FILL)
    centered(
        cx_train,
        y_mid,
        "Same model setup",
        "separate condition models\n30 epochs / round (uniform)\nYOLO11-pose n / s / m",
        body_fs=8.0,
    )

    box(cx_eval, y_mid, W, H + 0.15, edge=NEUTRAL_EDGE, fill=NEUTRAL_FILL)
    centered(
        cx_eval,
        y_mid,
        "Evaluation",
        "subject-disjoint internal val (989 fr.)\nused for best-checkpoint selection\npose mAP@[.50:.95] + kinematics",
        body_fs=7.6,
    )

    arrow(cx_pool + W / 2, y_mid, cx_sel - W / 2, y_mid)

    sel_r = cx_sel + W / 2
    lab_l = cx_lab - lab_w / 2
    for key, cy in lab_centers.items():
        rad = 0.0 if key == "B" else (0.10 if cy > y_mid else -0.10)
        arrow(sel_r, y_mid, lab_l, cy, color=ARROW_GREY, lw=1.6, rad=rad)

    lab_r = cx_lab + lab_w / 2
    train_l = cx_train - W / 2
    for key, cy in lab_centers.items():
        color = COND_STYLE[key]["color"]
        rad = 0.0 if key == "B" else (-0.10 if cy > y_mid else 0.10)
        arrow(lab_r, cy, train_l, y_mid, color=color, lw=2.0, rad=rad)

    arrow(cx_train + W / 2, y_mid, cx_eval - W / 2, y_mid)

    LOOP_C = "#6A5ACD"
    y_loop = 8.75
    y_eval_top = y_mid + (H + 0.15) / 2
    y_sel_top = y_mid + H / 2
    ax.plot([cx_eval, cx_eval], [y_eval_top, y_loop], color=LOOP_C, lw=1.8,
            ls=(0, (6, 4)), zorder=2)
    ax.plot([cx_eval, cx_sel], [y_loop, y_loop], color=LOOP_C, lw=1.8,
            ls=(0, (6, 4)), zorder=2)
    arrow(cx_sel, y_loop, cx_sel, y_sel_top, color=LOOP_C, lw=1.8, ls=(0, (6, 4)))
    ax.text(
        (cx_sel + cx_eval) / 2,
        y_loop + 0.20,
        "repeat per budget round:  0.05 → 0.10 → 0.20 → 0.40      (× 6 seeds)",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=LOOP_C,
        fontweight="bold",
        zorder=4,
    )

    save_fig(plt, args.out)
    plt.close(fig)
    print(f"[plot_method_diagram] wrote {args.out}")
    print(f"[plot_method_diagram] wrote {Path(args.out).with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
