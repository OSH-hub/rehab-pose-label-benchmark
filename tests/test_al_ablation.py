from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def _build_pool_fake(pool_imgs: list[str], rec_meta: dict) -> object:
    from rehabal_pose.al.base import Pool

    n = len(pool_imgs)
    K = 26
    rng = np.random.default_rng(42)
    preds = np.empty((n, K, 3), dtype=np.float64)
    preds[..., :2] = rng.uniform(0.0, 200.0, size=(n, K, 2))
    preds[..., 2] = rng.uniform(0.4, 0.9, size=(n, K))
    embeddings = rng.normal(size=(n, 8))
    clip_ids = [f"clip{i // 4}" for i in range(n)]
    frame_ids = [i % 4 for i in range(n)]
    metadata = {
        "disease": ["stroke" if i % 2 == 0 else "frozen_shoulder" for i in range(n)],
        "exercise": ["ex_a" if i % 3 == 0 else "ex_b" for i in range(n)],
        "camera_id": ["camA" if i % 4 < 2 else "camB" for i in range(n)],
    }
    return Pool(
        sample_ids=[Path(p).stem for p in pool_imgs],
        predictions=preds,
        embeddings=embeddings,
        clip_ids=clip_ids,
        frame_ids=frame_ids,
        metadata=metadata,
    )


def _make_fake_pool_imgs(n: int, tmp_path: Path) -> list[str]:
    d = tmp_path / "images" / "train"
    d.mkdir(parents=True, exist_ok=True)
    imgs = []
    for i in range(n):
        p = d / f"frame_{i:04d}.jpg"
        p.touch()
        imgs.append(str(p))
    return sorted(imgs)


def _fake_train_eval_factory(counter: dict):
    def _fake(selected_imgs, out_dir, epochs, imgsz, seed, batch=16):
        counter["calls"] = counter.get("calls", 0) + 1
        return {
            "weights": str(Path(out_dir) / "fake_best.pt"),
            "map50": 0.3 + 0.01 * counter["calls"],
            "map50_95": 0.2 + 0.01 * counter["calls"],
        }
    return _fake


def _fake_manifest_df(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "sample_id": f"frame_{i:04d}",
            "subject_id": f"subj_{i % 4:02d}",
            "video_id": f"vid_{i:04d}",
            "exercise": f"ex{i % 2}",
            "disease": "stroke",
            "camera_id": "camA",
        }
        for i in range(n)
    ])


def _fake_rec_meta(n: int = 20) -> dict:
    return {
        f"vid_{i:04d}": {
            "disease": "stroke",
            "exercise": f"ex{i % 2}",
            "camera_id": "camA",
            "subject_id": f"subj_{i % 4:02d}",
        }
        for i in range(n)
    }


_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def test_ablation_variant_count_and_names():
    import al_ablation

    variants = al_ablation.build_ablation_variants()
    assert len(variants) == 7
    assert "rehabal_full" in variants
    for term in al_ablation.ABLATION_TERMS:
        assert f"rehabal_no_{term}" in variants


def test_rehabal_full_uses_default_weights():
    from rehabal_pose.al import DEFAULT_WEIGHTS

    import al_ablation

    variants = al_ablation.build_ablation_variants()
    full = variants["rehabal_full"]
    for key in DEFAULT_WEIGHTS:
        assert key in full.weights, f"rehabal_full missing weight key {key!r}"
        assert full.weights[key] > 0.0, (
            f"rehabal_full weight for {key!r} should be > 0; got {full.weights[key]}"
        )
    assert abs(sum(full.weights.values()) - 1.0) < 1e-9


def test_leave_one_out_variants_zero_target_term():
    import al_ablation

    variants = al_ablation.build_ablation_variants()
    for term in al_ablation.ABLATION_TERMS:
        key = f"rehabal_no_{term}"
        strat = variants[key]
        assert strat.weights[term] == pytest.approx(0.0), (
            f"{key}: expected {term} weight == 0; got {strat.weights[term]}"
        )
        for other in al_ablation.ABLATION_TERMS:
            if other == term:
                continue
            assert strat.weights[other] > 0.0, (
                f"{key}: term {other!r} should be > 0 after ablating {term!r}; "
                f"got {strat.weights[other]}"
            )
        assert abs(sum(strat.weights.values()) - 1.0) < 1e-9


def _ablation_patches(pool_imgs, tmp_path):
    manifest_df = _fake_manifest_df()
    rec_meta = _fake_rec_meta()

    def _fake_score_pool(imgs, wpath, rmeta):
        return _build_pool_fake(imgs, rmeta)

    patches = [
        patch("al_ablation.glob.glob", return_value=pool_imgs),
        patch("al_ablation.ManifestBuilder.load", return_value=manifest_df),
        patch("al_ablation._recording_meta", return_value=rec_meta),
        patch("al_ablation._score_pool_ablation", side_effect=_fake_score_pool),
        patch("al_ablation.DATASET", tmp_path / "yolo_dataset"),
        patch("al_ablation.MANIFEST", tmp_path / "manifest.parquet"),
    ]
    return patches


def test_main_smoke_runs_end_to_end(tmp_path):
    import al_ablation

    pool_imgs = _make_fake_pool_imgs(n=20, tmp_path=tmp_path)
    out_dir = tmp_path / "ablation_out"
    counter: dict = {}
    fake_te = _fake_train_eval_factory(counter)

    patches = _ablation_patches(pool_imgs, tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        ret = al_ablation.main(
            ["--smoke", "--out", str(out_dir)],
            train_eval=fake_te,
        )

    assert ret == 0, "main() returned non-zero exit code"
    assert (out_dir / "curve.csv").exists(), "curve.csv not written"
    assert (out_dir / "summary.json").exists(), "summary.json not written"
    assert counter.get("calls", 0) > 0, "fake train_eval was never called"


def test_curve_csv_schema(tmp_path):
    import al_ablation

    pool_imgs = _make_fake_pool_imgs(n=20, tmp_path=tmp_path)
    out_dir = tmp_path / "ablation_schema"

    patches = _ablation_patches(pool_imgs, tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        al_ablation.main(
            ["--smoke", "--out", str(out_dir)],
            train_eval=_fake_train_eval_factory({}),
        )

    df = pd.read_csv(out_dir / "curve.csv")
    required_cols = {
        "strategy", "budget", "seed", "n_labeled", "pose_map50", "pose_map50_95"
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"curve.csv missing columns: {missing}"
    assert len(df) > 0, "curve.csv is empty"
    assert df["strategy"].isin(
        ["rehabal_full"] + [f"rehabal_no_{t}" for t in al_ablation.ABLATION_TERMS]
    ).all()


def test_summary_json_structure(tmp_path):
    import al_ablation

    pool_imgs = _make_fake_pool_imgs(n=20, tmp_path=tmp_path)
    out_dir = tmp_path / "ablation_summary"

    patches = _ablation_patches(pool_imgs, tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        al_ablation.main(
            ["--smoke", "--out", str(out_dir)],
            train_eval=_fake_train_eval_factory({}),
        )

    with open(out_dir / "summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)

    assert len(summary) == 2
    for variant_name, stats in summary.items():
        assert "auc_map50_95_mean" in stats, (
            f"{variant_name}: missing auc_map50_95_mean"
        )
        assert "auc_map50_95_std" in stats, (
            f"{variant_name}: missing auc_map50_95_std"
        )
        assert "budget_stats" in stats, (
            f"{variant_name}: missing budget_stats"
        )
        for budget_key, bstats in stats["budget_stats"].items():
            for field in ("mean", "std", "n"):
                assert field in bstats, (
                    f"{variant_name} budget {budget_key}: missing '{field}' in budget_stats"
                )


def test_resume_skips_done_cells(tmp_path):
    import al_ablation

    pool_imgs = _make_fake_pool_imgs(n=20, tmp_path=tmp_path)
    out_dir = tmp_path / "ablation_resume"

    counter1: dict = {}
    patches = _ablation_patches(pool_imgs, tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        al_ablation.main(
            ["--smoke", "--out", str(out_dir)],
            train_eval=_fake_train_eval_factory(counter1),
        )

    calls_run1 = counter1.get("calls", 0)
    assert calls_run1 > 0, "First run made no training calls"

    counter2: dict = {}
    patches2 = _ablation_patches(pool_imgs, tmp_path)
    with patches2[0], patches2[1], patches2[2], patches2[3], patches2[4], patches2[5]:
        al_ablation.main(
            ["--smoke", "--out", str(out_dir)],
            train_eval=_fake_train_eval_factory(counter2),
        )

    calls_run2 = counter2.get("calls", 0)
    assert calls_run2 == 0, (
        f"Second run should skip all done cells but made {calls_run2} training calls"
    )


def test_no_heavy_imports_from_ablation_module():
    check_script = (
        "import sys, pathlib; "
        f"sys.path.insert(0, r'{_SCRIPTS_DIR}'); "
        "sys.path.insert(0, r'src'); "
        "import al_ablation; "  # noqa: implicit continuation
        "bad = [m for m in ('torch','ultralytics','cv2','matplotlib') if m in sys.modules]; "
        "sys.exit(0) if not bad else (print('HEAVY:', bad) or sys.exit(1))"
    )
    result = subprocess.run(
        [sys.executable, "-c", check_script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"al_ablation module-load imported heavy dep(s):\n"
        f"stdout: {result.stdout.strip()}\n"
        f"stderr: {result.stderr.strip()}"
    )


def test_resume_restores_labeled_set_exactly(tmp_path):
    import al_ablation

    pool_imgs = _make_fake_pool_imgs(n=20, tmp_path=tmp_path)
    out_dir = tmp_path / "ablation_labeled_restore"

    counter1: dict = {"calls": 0}

    trained_labeled: list[list[int]] = []

    def _recording_te(selected_imgs, out_dir_arg, epochs, imgsz, seed, batch=16):
        counter1["calls"] += 1
        trained_labeled.append(list(selected_imgs))
        return {
            "weights": str(Path(out_dir_arg) / "fake_best.pt"),
            "map50": 0.3 + 0.01 * counter1["calls"],
            "map50_95": 0.2 + 0.01 * counter1["calls"],
        }

    patches = _ablation_patches(pool_imgs, tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        ret = al_ablation.main(
            ["--smoke", "--out", str(out_dir)],
            train_eval=_recording_te,
        )
    assert ret == 0

    csv_df = pd.read_csv(out_dir / "curve.csv")
    for _, row in csv_df.iterrows():
        tag = f"{row['strategy']}_b{row['budget']:g}_s{int(row['seed'])}"
        idx_json = out_dir / tag / "labeled_idx.json"
        assert idx_json.exists(), f"labeled_idx.json missing for {tag}"
        with idx_json.open(encoding="utf-8") as fh:
            saved_idx = json.load(fh)
        assert isinstance(saved_idx, list), f"{tag}: labeled_idx.json is not a list"
        assert len(saved_idx) == int(row["n_labeled"]), (
            f"{tag}: saved {len(saved_idx)} indices but n_labeled={row['n_labeled']}"
        )
        assert all(0 <= i < len(pool_imgs) for i in saved_idx), (
            f"{tag}: out-of-range index in labeled_idx.json"
        )

    full_b1_tag = f"rehabal_full_b0.1_s0"
    keep_mask = ~(
        (csv_df["strategy"] == "rehabal_full") &
        (csv_df["budget"].astype(float) == 0.1) &
        (csv_df["seed"].astype(int) == 0)
    )
    csv_df[keep_mask].to_csv(out_dir / "curve.csv", index=False)

    first_budget_idx_json = out_dir / "rehabal_full_b0.05_s0" / "labeled_idx.json"
    with first_budget_idx_json.open(encoding="utf-8") as fh:
        expected_labeled_after_b1 = json.load(fh)

    resume_labeled_received: list = []

    def _resume_te(selected_imgs, out_dir_arg, epochs, imgsz, seed, batch=16):
        resume_labeled_received.extend(selected_imgs)
        return {
            "weights": str(Path(out_dir_arg) / "fake_best.pt"),
            "map50": 0.35,
            "map50_95": 0.25,
        }

    counter2: dict = {"calls": 0}

    def _resume_te_counted(selected_imgs, out_dir_arg, epochs, imgsz, seed, batch=16):
        counter2["calls"] += 1
        resume_labeled_received.extend(selected_imgs)
        return {
            "weights": str(Path(out_dir_arg) / "fake_best.pt"),
            "map50": 0.35,
            "map50_95": 0.25,
        }

    patches2 = _ablation_patches(pool_imgs, tmp_path)
    with patches2[0], patches2[1], patches2[2], patches2[3], patches2[4], patches2[5]:
        al_ablation.main(
            ["--smoke", "--out", str(out_dir)],
            train_eval=_resume_te_counted,
        )

    assert counter2["calls"] == 1, (
        f"Expected 1 re-run cell on resume, got {counter2['calls']}"
    )

    b05_set = set(expected_labeled_after_b1)
    rerun_imgs_set = set(resume_labeled_received)
    b05_stems = {Path(pool_imgs[i]).stem for i in b05_set}
    rerun_stems = {Path(p).stem for p in resume_labeled_received}
    assert b05_stems.issubset(rerun_stems), (
        f"Resume did not preserve the prior round's labeled set.\n"
        f"Missing stems: {b05_stems - rerun_stems}"
    )
