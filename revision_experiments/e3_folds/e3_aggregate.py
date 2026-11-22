r"""E3 aggregation + analysis (CPU only, safe to run any time; partial data OK).

Produces under rehabal_work\e3_folds\ :
  fold{k}/clinician_curve.csv   (same format as the original master; fold 5 = rows of
                                 the existing main grid, seeds 0-1, A / B sig0 / ref)
  e3_clinician_curve_all.csv    (all folds, with a fold column)
  e3_eval.csv                   (merged re-evaluation rows: best.pt AND last.pt, every round)
  e3_summary.txt                (per-fold and pooled tables, AUC, gap closure, best vs last)

Usage: python rehabal_work\e3_aggregate.py
"""
import json, os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"RehabAL-pose")
WORK = Path(r"rehabal_work\e3_folds")
ORIG = REPO / "outputs" / "experiments" / "clinician_i4_uniform30"
BUDGETS = [0.05, 0.10, 0.20, 0.40]
CONDS = ["A", "B", "ref"]
SEEDS = [0, 1]
FOLD_SUBJ = {1: "1,2", 2: "3,4", 3: "5,6", 4: "7,8", 5: "9,10"}
CURVE_COLS = ["condition", "seed", "sigma", "budget", "n_labeled", "map50", "map50_95",
              "n_proposed", "n_accepted", "n_rejected", "n_corrected", "clinician"]


def tag_of(cond, seed):
    return f"{cond}_s{seed}" + ("_sig0" if cond == "B" else "")


def atomic_csv(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def rows_from_curve(curve_csv, cond, seed):
    df = pd.read_csv(curve_csv)
    if len(df) != len(BUDGETS):
        return None
    rows = []
    for i, rec in df.reset_index(drop=True).iterrows():
        rows.append({"condition": cond, "seed": seed, "sigma": 0.0 if cond == "B" else "",
                     "budget": BUDGETS[i], "n_labeled": int(rec["n_labeled"]),
                     "map50": rec["map50"], "map50_95": rec["map50_95"],
                     "n_proposed": int(rec["n_proposed"]) if "n_proposed" in rec and pd.notna(rec.get("n_proposed")) else None,
                     "n_accepted": int(rec["n_accepted"]), "n_rejected": int(rec["n_rejected"]),
                     "n_corrected": int(rec["n_corrected"]), "clinician": rec["clinician"]})
    return rows


def build_curves():
    all_rows = []
    status = {}
    for k in (1, 2, 3, 4):
        rows = []
        done = []
        for seed in SEEDS:
            for cond in CONDS:
                sd = WORK / f"fold{k}" / tag_of(cond, seed)
                if (sd / "completion.json").is_file() and (sd / "curve.csv").is_file():
                    r = rows_from_curve(sd / "curve.csv", cond, seed)
                    if r:
                        rows.extend(r)
                        done.append(tag_of(cond, seed))
        status[k] = done
        if rows:
            df = pd.DataFrame(rows, columns=CURVE_COLS)
            atomic_csv(df, WORK / f"fold{k}" / "clinician_curve.csv")
            df.insert(0, "fold", k)
            all_rows.append(df)
    # fold 5 = existing main grid
    m = ORIG / "clinician_curve.csv"
    if m.is_file():
        df = pd.read_csv(m)
        sig = pd.to_numeric(df["sigma"], errors="coerce").fillna(0.0)
        df = df[df["condition"].isin(CONDS) & df["seed"].isin(SEEDS) & (sig == 0.0)].copy()
        df = df.sort_values(["seed", "condition", "budget"]).reset_index(drop=True)
        (WORK / "fold5").mkdir(parents=True, exist_ok=True)
        atomic_csv(df[CURVE_COLS], WORK / "fold5" / "clinician_curve.csv")
        status[5] = sorted({tag_of(c, s) for c, s in zip(df["condition"], df["seed"])})
        df = df[CURVE_COLS].copy()
        df.insert(0, "fold", 5)
        all_rows.append(df)
    if all_rows:
        allc = pd.concat(all_rows, ignore_index=True)
        atomic_csv(allc, WORK / "e3_clinician_curve_all.csv")
    else:
        allc = pd.DataFrame()
    return allc, status


def merge_eval():
    parts = []
    for p in sorted(WORK.glob("fold*/*/e3_eval_rows.csv")):
        try:
            parts.append(pd.read_csv(p))
        except Exception:  # noqa: BLE001
            pass
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["fold", "condition", "seed", "round", "ckpt"], keep="last")
    df = df.sort_values(["fold", "condition", "seed", "round", "ckpt"]).reset_index(drop=True)
    lead = ["fold", "condition", "seed", "round", "budget", "ckpt", "map50", "map50_95", "weights"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    atomic_csv(df, WORK / "e3_eval.csv")
    return df


def ms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "   --   "
    if x.size == 1:
        return f"{x[0]:.4f}      "
    return f"{x.mean():.4f}±{x.std(ddof=1):.4f}"


def nauc(bud, y):
    bud = np.asarray(bud, float)
    y = np.asarray(y, float)
    o = np.argsort(bud)
    bud, y = bud[o], y[o]
    if len(bud) < 2 or not np.isfinite(y).all():
        return np.nan
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(y, bud) / (bud[-1] - bud[0]))


def complete_units(ev, metric):
    """(fold, seed, condition) units with all 4 rounds -> dict and per-unit AUC."""
    g = ev.groupby(["fold", "seed", "condition"])
    units = {}
    for key, sub in g:
        if len(sub) == 4:
            units[key] = sub.sort_values("budget")
    return units


def section_tables(out, ev, metric, label):
    out.append(f"\n==== {label}: per-fold tables, mean±std over seeds (n_seed per cell in parentheses) ====")
    for k in sorted(ev["fold"].unique()):
        sub = ev[ev.fold == k]
        out.append(f"\n-- fold {k} (val subjects {FOLD_SUBJ.get(k)}) --")
        out.append("cond   " + "  ".join(f"b={b:<13}" for b in BUDGETS))
        for c in CONDS:
            cells = []
            for b in BUDGETS:
                v = sub[(sub.condition == c) & (np.isclose(sub.budget, b))][metric].values
                cells.append(f"{ms(v)}({len(v)})")
            out.append(f"{c:<6} " + "  ".join(cells))


def section_pooled(out, ev, metric, label):
    out.append(f"\n==== {label}: POOLED ====")
    fm = ev.groupby(["fold", "condition", "budget"])[metric].mean().reset_index()  # fold-level means
    nf = fm["fold"].nunique()
    out.append(f"(a) fold as resampling unit: mean±std over {nf} fold-means (each fold-mean = mean over its seeds)")
    out.append("cond   " + "  ".join(f"b={b:<13}" for b in BUDGETS))
    for c in CONDS:
        cells = [ms(fm[(fm.condition == c) & np.isclose(fm.budget, b)][metric].values) for b in BUDGETS]
        out.append(f"{c:<6} " + "  ".join(cells))
    out.append(f"(b) all fold-seed units pooled: mean±std over up to {ev.groupby(['fold','seed']).ngroups} units")
    out.append("cond   " + "  ".join(f"b={b:<13}" for b in BUDGETS))
    for c in CONDS:
        cells = [ms(ev[(ev.condition == c) & np.isclose(ev.budget, b)][metric].values) for b in BUDGETS]
        out.append(f"{c:<6} " + "  ".join(cells))
    # gap closure per budget on fold-means
    out.append("(c) B gap-closure of the A->ref gap per budget, computed per fold on fold-means: (B-A)/(ref-A); mean±std over folds")
    rows = []
    for k in sorted(fm.fold.unique()):
        f = fm[fm.fold == k]
        for b in BUDGETS:
            a = f[(f.condition == "A") & np.isclose(f.budget, b)][metric]
            bb = f[(f.condition == "B") & np.isclose(f.budget, b)][metric]
            r = f[(f.condition == "ref") & np.isclose(f.budget, b)][metric]
            if len(a) and len(bb) and len(r):
                gap = float(r.iloc[0] - a.iloc[0])
                rows.append({"fold": k, "budget": b, "A": a.iloc[0], "B": bb.iloc[0], "ref": r.iloc[0],
                             "gap_ref_A": gap, "closure": (bb.iloc[0] - a.iloc[0]) / gap if abs(gap) > 1e-9 else np.nan})
    gc = pd.DataFrame(rows)
    if len(gc):
        for b in BUDGETS:
            if b == BUDGETS[0]:
                out.append(f"  b={b:<5} closure n/a (round 0 = shared random GT init, identical across conditions)")
                continue
            s = gc[np.isclose(gc.budget, b)]
            out.append(f"  b={b:<5} closure {ms(s.closure.values)}  (ref-A gap {ms(s.gap_ref_A.values)}, "
                       f"per-fold closure: {', '.join(f'F{int(r.fold)}:{r.closure:+.2f}' for r in s.itertuples())})")
    # AUC
    out.append("(d) normalized AUC over budgets 0.05..0.40 (trapezoid / 0.35) per (fold, seed, condition)")
    units = complete_units(ev, metric)
    au = pd.DataFrame([{"fold": k[0], "seed": k[1], "condition": k[2],
                        "auc": nauc(u.budget, u[metric])} for k, u in units.items()])
    if len(au):
        fa = au.groupby(["fold", "condition"])["auc"].mean().reset_index()
        out.append("    cond   fold-level mean±std (n folds)   all-units mean±std (n units)")
        for c in CONDS:
            v1 = fa[fa.condition == c].auc.values
            v2 = au[au.condition == c].auc.values
            out.append(f"    {c:<6} {ms(v1)} ({len(v1)})           {ms(v2)} ({len(v2)})")
        piv = fa.pivot(index="fold", columns="condition", values="auc")
        if all(c in piv.columns for c in CONDS):
            piv = piv.dropna()
            out.append("    per-fold AUC:  " + "  ".join(
                f"F{int(i)}: A={r.A:.4f} B={r.B:.4f} ref={r.ref:.4f}" for i, r in piv.iterrows()))
            dBA, drA, drB = piv.B - piv.A, piv.ref - piv.A, piv.ref - piv.B
            clos = (piv.B - piv.A) / (piv.ref - piv.A)
            out.append(f"    AUC diffs over folds: B-A {ms(dBA)}  ref-A {ms(drA)}  ref-B {ms(drB)}  "
                       f"| B>A in {int((dBA > 0).sum())}/{len(dBA)} folds, ref>B in {int((drB > 0).sum())}/{len(drB)}")
            out.append(f"    AUC gap-closure (B-A)/(ref-A): {ms(clos)}  per fold: "
                       + ", ".join(f"F{int(i)}:{v:+.2f}" for i, v in clos.items()))
            if len(piv) >= 3:
                try:
                    from scipy import stats
                    for name, d in (("B-A", dBA), ("ref-A", drA), ("ref-B", drB)):
                        t = stats.ttest_1samp(d.values, 0.0)  # paired differences vs zero
                        try:
                            wl = stats.wilcoxon(d.values)
                            wp = f"{wl.pvalue:.3f}"
                        except Exception:  # noqa: BLE001
                            wp = "n/a"
                        out.append(f"    paired over folds (n={len(d)}): {name}: t={t.statistic:.2f} p={t.pvalue:.3f}; "
                                   f"Wilcoxon p={wp}  [low power at n={len(d)}]")
                except Exception as exc:  # noqa: BLE001
                    out.append(f"    (scipy tests unavailable: {exc})")
    return au


def section_best_last(out, ev, metric, label):
    out.append(f"\n==== {label}: best.pt vs last.pt (same val split; best.pt was selected on it) ====")
    piv = ev.pivot_table(index=["fold", "seed", "condition", "budget"], columns="ckpt", values=metric).reset_index()
    if "best" not in piv.columns or "last" not in piv.columns:
        out.append("  (need both ckpts)")
        return
    piv = piv.dropna(subset=["best", "last"])
    piv["diff"] = piv["best"] - piv["last"]
    out.append("  mean±std of (best - last) over fold-seed units, per condition x budget:")
    out.append("cond   " + "  ".join(f"b={b:<13}" for b in BUDGETS))
    for c in CONDS:
        cells = [ms(piv[(piv.condition == c) & np.isclose(piv.budget, b)]["diff"].values) for b in BUDGETS]
        out.append(f"{c:<6} " + "  ".join(cells))
    out.append(f"  overall best-last: {ms(piv['diff'].values)} (n={len(piv)}); "
               f"best>=last in {int((piv['diff'] >= 0).sum())}/{len(piv)}")
    out.append("  pooled table with last.pt (fold-level means, mean±std over folds):")
    lm = ev[ev.ckpt == "last"].groupby(["fold", "condition", "budget"])[metric].mean().reset_index()
    out.append("cond   " + "  ".join(f"b={b:<13}" for b in BUDGETS))
    for c in CONDS:
        cells = [ms(lm[(lm.condition == c) & np.isclose(lm.budget, b)][metric].values) for b in BUDGETS]
        out.append(f"{c:<6} " + "  ".join(cells))


def analysis(allc, ev, status):
    out = []
    out.append("E3: 5-fold subject-independent re-validation of A / B(sigma=0) / ref")
    out.append(f"generated {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    out.append("folds: F1={1,2} F2={3,4} F3={5,6} F4={7,8} F5={9,10} (F5 = existing main grid clinician_i4_uniform30)")
    out.append("protocol: temporal AL, budgets 0.05/0.10/0.20/0.40 of the fold's train pool, 30 ep every round, imgsz 416, batch 16, yolo11n-pose, seeds 0-1")
    out.append("val sizes: F1=2406 F2=1869 F3=1812 F4=2280 F5=989 frames (per-subject frame counts differ: subject 10 has only 55 frames)")
    out.append("\n== completion status ==")
    for k in (1, 2, 3, 4, 5):
        out.append(f"  fold{k}: {len(status.get(k, []))}/6 sessions: {status.get(k, [])}")
    out.append(f"  e3_eval.csv rows: {len(ev)} (target 5 folds x 6 sessions x 4 rounds x 2 ckpts = 240)")
    if len(ev) == 0:
        return "\n".join(out)
    # sanity: re-evaluated best.pt vs training-time curve metric
    if len(allc):
        m = ev[ev.ckpt == "best"].merge(allc, on=["fold", "condition", "seed", "budget"],
                                        suffixes=("_reeval", "_train"))
        if len(m):
            d95 = (m.map50_95_reeval - m.map50_95_train).abs()
            d50 = (m.map50_reeval - m.map50_train).abs()
            out.append(f"\n== sanity: re-evaluated best.pt vs training-time curve (n={len(m)}): "
                       f"max|dmAP50-95|={d95.max():.4f} mean={d95.mean():.4f}; max|dmAP50|={d50.max():.4f}")
    for metric, label in (("map50_95", "pose mAP50-95"), ("map50", "pose mAP50")):
        for ck in ("best", "last"):
            sub = ev[ev.ckpt == ck]
            if len(sub) == 0:
                continue
            section_tables(out, sub, metric, f"{label} [{ck}.pt]")
            section_pooled(out, sub, metric, f"{label} [{ck}.pt]")
        section_best_last(out, ev, metric, label)
    out.append("\n== caveats ==")
    out.append("* Subjects 1-8 were the training pool while the original protocol/hyper-parameters were developed; folds 1-4 are an internal re-validation, not an untouched test set.")
    out.append("* best.pt is selected on the same held-out split it is reported on; last.pt rows remove that selection step.")
    out.append("* Only 2 seeds per fold; fold-level std (n=5) is the honest spread; paired tests over 5 folds have low power.")
    out.append("* Fold val sizes and subject composition differ (F5 is dominated by subject 9), so absolute levels are not comparable across folds; compare conditions within fold.")
    return "\n".join(out)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    allc, status = build_curves()
    ev = merge_eval()
    txt = analysis(allc, ev, status)
    (WORK / "e3_summary.txt").write_text(txt, encoding="utf-8")
    (WORK / "e3_status.json").write_text(json.dumps({"sessions": status, "eval_rows": int(len(ev))}, indent=2), encoding="utf-8")
    print(txt[:1500])
    print("E3_AGGREGATE_DONE", flush=True)


if __name__ == "__main__":
    main()
