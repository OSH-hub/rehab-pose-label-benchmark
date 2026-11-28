from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import pytest


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


_CALL_LOG: list[tuple] = []


def _make_fake_train_eval(*, base_map: float = 0.10, slope: float = 0.0005):
    def fake_train_eval(
        selected_imgs: list,
        out_dir: Path,
        epochs: int,
        imgsz: int,
        seed: int,
        batch: int,
    ) -> dict:
        n = len(selected_imgs)
        map_val = min(base_map + slope * n, 1.0)
        out_dir.mkdir(parents=True, exist_ok=True)
        w_dir = out_dir / "run" / "train" / "weights"
        w_dir.mkdir(parents=True, exist_ok=True)
        (w_dir / "best.pt").write_text("fake", encoding="utf-8")
        _CALL_LOG.append((str(out_dir), n))
        return {
            "map50": map_val * 0.9,
            "map50_95": map_val,
            "weights": str(w_dir / "best.pt"),
        }

    return fake_train_eval


class _FakeStrategy:
    def select(self, pool, k: int, candidates: Optional[list] = None, **_):
        cands = candidates if candidates is not None else list(range(len(pool)))
        return list(cands[:k])


def _fake_strategy_factory(name: str) -> _FakeStrategy:
    return _FakeStrategy()


def _make_fake_pool(n_images: int = 20):
    from rehabal_pose.al.base import Pool  # noqa: PLC0415

    K = 26
    rng = np.random.default_rng(42)
    preds = np.empty((n_images, K, 3), dtype=np.float64)
    preds[..., :2] = rng.uniform(0.0, 200.0, size=(n_images, K, 2))
    preds[..., 2] = rng.uniform(0.5, 1.0, size=(n_images, K))
    embeds = rng.normal(size=(n_images, 8))
    return Pool(
        sample_ids=[f"s{i:04d}" for i in range(n_images)],
        predictions=preds,
        embeddings=embeds,
    )


def _make_fake_pool_factory(n_images: int = 20):
    def _factory(pool_imgs, weights, rec_meta):
        return _make_fake_pool(len(pool_imgs))
    return _factory


def _patch_runner(monkeypatch, tmp_path: Path, n_images: int = 20):
    import run_paper_experiments as rpe

    img_dir = tmp_path / "rehab24_6_yolo" / "images" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        (img_dir / f"frame_{i:04d}.jpg").write_bytes(b"")

    import al_budget_curve as abc_mod  # noqa: PLC0415
    monkeypatch.setattr(abc_mod, "DATASET", tmp_path / "rehab24_6_yolo")
    monkeypatch.setattr(rpe, "DATASET", tmp_path / "rehab24_6_yolo")

    from rehabal_pose.data import ManifestBuilder  # noqa: PLC0415
    tiny_manifest = pd.DataFrame({"video_id": ["v0"], "subject_id": ["s0"],
                                   "exercise": ["ex1"], "disease": ["stroke"],
                                   "camera_id": ["c17"]})

    monkeypatch.setattr(ManifestBuilder, "load", staticmethod(lambda _: tiny_manifest))

    manifest_path = tmp_path / "manifest.parquet"
    manifest_path.write_bytes(b"")
    monkeypatch.setattr(abc_mod, "MANIFEST", manifest_path)
    monkeypatch.setattr(rpe, "MANIFEST", manifest_path)


def test_correct_number_of_cells(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, tmp_path, n_images=30)

    import run_paper_experiments as rpe  # noqa: PLC0415

    out_dir = tmp_path / "out"
    strategies = ["random", "uncertainty"]
    budgets = [0.1, 0.3, 1.0]
    seeds = [0, 1]

    ret = rpe.main(
        argv=[
            "--strategies", *strategies,
            "--budgets", *[str(b) for b in budgets],
            "--seeds", *[str(s) for s in seeds],
            "--out", str(out_dir),
            "--epochs", "1",
        ],
        train_eval=_make_fake_train_eval(),
        strategy_factory=_fake_strategy_factory,
        pool_factory=_make_fake_pool_factory(30),
    )

    assert ret == 0, "main() should return 0"
    csv_path = out_dir / "curve.csv"
    assert csv_path.is_file(), "curve.csv must be created"
    df = pd.read_csv(csv_path)
    expected = len(strategies) * len(budgets) * len(seeds)
    assert len(df) == expected, (
        f"Expected {expected} rows, got {len(df)}"
    )


def test_curve_csv_schema(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, tmp_path, n_images=20)

    import run_paper_experiments as rpe  # noqa: PLC0415

    out_dir = tmp_path / "out_schema"
    ret = rpe.main(
        argv=[
            "--strategies", "random",
            "--budgets", "0.1", "0.5",
            "--seeds", "0",
            "--out", str(out_dir),
            "--epochs", "1",
        ],
        train_eval=_make_fake_train_eval(),
        strategy_factory=_fake_strategy_factory,
        pool_factory=_make_fake_pool_factory(20),
    )

    assert ret == 0
    df = pd.read_csv(out_dir / "curve.csv")

    required_cols = {"strategy", "budget", "seed", "n_labeled",
                     "pose_map50", "pose_map50_95"}
    assert required_cols.issubset(set(df.columns)), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )
    assert (df["strategy"] == "random").all()
    assert df["budget"].dtype in (float, np.float64, np.float32)
    assert df["n_labeled"].dtype in (int, np.int64, np.int32, np.float64)
    assert df["pose_map50"].notna().all()
    assert df["pose_map50_95"].notna().all()


def test_summary_json_structure(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, tmp_path, n_images=30)

    import run_paper_experiments as rpe  # noqa: PLC0415

    out_dir = tmp_path / "out_summary"
    strategies = ["random", "rehabal"]
    budgets = [0.05, 0.20, 1.00]
    seeds = [0, 1, 2]

    ret = rpe.main(
        argv=[
            "--strategies", *strategies,
            "--budgets", *[str(b) for b in budgets],
            "--seeds", *[str(s) for s in seeds],
            "--out", str(out_dir),
            "--epochs", "1",
        ],
        train_eval=_make_fake_train_eval(),
        strategy_factory=_fake_strategy_factory,
        pool_factory=_make_fake_pool_factory(30),
    )

    assert ret == 0
    json_path = out_dir / "summary.json"
    assert json_path.is_file(), "summary.json must be created"

    with json_path.open(encoding="utf-8") as fh:
        summary = json.load(fh)

    for strat in strategies:
        assert strat in summary, f"strategy {strat} missing from summary.json"
        info = summary[strat]

        assert "auc_map50_95" in info, "auc_map50_95 key missing"
        assert "mean" in info["auc_map50_95"] and "std" in info["auc_map50_95"]
        auc_mean = info["auc_map50_95"]["mean"]
        auc_std = info["auc_map50_95"]["std"]
        assert auc_mean is not None and auc_mean == auc_mean, "AUC mean must not be NaN"
        assert auc_std is not None and auc_std >= 0.0, "AUC std must be >= 0"

        assert "budgets" in info, "budgets key missing"
        for b in budgets:
            bkey = float(b)
            matching = [k for k in info["budgets"] if abs(float(k) - bkey) < 1e-9]
            assert matching, f"budget {b} missing from summary.json[{strat}][budgets]"
            binfo = info["budgets"][matching[0]]
            assert "map50_95_mean" in binfo
            assert "map50_95_std" in binfo
            assert binfo["map50_95_mean"] is not None
            assert binfo["map50_95_std"] is not None and binfo["map50_95_std"] >= 0.0


def test_auc_per_seed_not_of_mean(monkeypatch, tmp_path):
    from rehabal_pose.metrics.learning_curve import area_under_learning_curve  # noqa: PLC0415

    _patch_runner(monkeypatch, tmp_path, n_images=40)

    import run_paper_experiments as rpe  # noqa: PLC0415

    out_dir = tmp_path / "out_auc"
    strategies = ["random"]
    budgets = [0.1, 0.3, 0.6, 1.0]
    seeds = [0, 1, 2]

    ret = rpe.main(
        argv=[
            "--strategies", *strategies,
            "--budgets", *[str(b) for b in budgets],
            "--seeds", *[str(s) for s in seeds],
            "--out", str(out_dir),
            "--epochs", "1",
        ],
        train_eval=_make_fake_train_eval(base_map=0.10, slope=0.002),
        strategy_factory=_fake_strategy_factory,
        pool_factory=_make_fake_pool_factory(40),
    )

    assert ret == 0
    df = pd.read_csv(out_dir / "curve.csv")

    strat_df = df[df["strategy"] == "random"]
    per_seed_aucs = []
    for seed in seeds:
        seed_sub = strat_df[strat_df["seed"] == seed].sort_values("budget")
        bs = seed_sub["budget"].astype(float).tolist()
        vs = seed_sub["pose_map50_95"].astype(float).tolist()
        if len(bs) >= 2:
            per_seed_aucs.append(area_under_learning_curve(bs, vs, normalize=True))
        else:
            per_seed_aucs.append(float("nan"))

    expected_mean = float(np.nanmean(per_seed_aucs))
    expected_std = float(np.nanstd(per_seed_aucs, ddof=0))

    with (out_dir / "summary.json").open(encoding="utf-8") as fh:
        summary = json.load(fh)

    actual_mean = summary["random"]["auc_map50_95"]["mean"]
    actual_std = summary["random"]["auc_map50_95"]["std"]

    assert abs(actual_mean - expected_mean) < 1e-9, (
        f"AUC mean mismatch: expected {expected_mean:.6f}, got {actual_mean:.6f}"
    )
    assert abs(actual_std - expected_std) < 1e-9, (
        f"AUC std mismatch: expected {expected_std:.6f}, got {actual_std:.6f}"
    )


def test_resume_skips_done_cells(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, tmp_path, n_images=20)

    import run_paper_experiments as rpe  # noqa: PLC0415

    out_dir = tmp_path / "out_resume"
    common_argv = [
        "--strategies", "random", "uncertainty",
        "--budgets", "0.2", "0.5",
        "--seeds", "0",
        "--out", str(out_dir),
        "--epochs", "1",
    ]

    call_count_first: list[int] = [0]
    call_count_second: list[int] = [0]

    def counting_train_eval_1(selected_imgs, out_dir, epochs, imgsz, seed, batch):
        call_count_first[0] += 1
        return _make_fake_train_eval()(selected_imgs, out_dir, epochs, imgsz, seed, batch)

    def counting_train_eval_2(selected_imgs, out_dir, epochs, imgsz, seed, batch):
        call_count_second[0] += 1
        return _make_fake_train_eval()(selected_imgs, out_dir, epochs, imgsz, seed, batch)

    _pool_factory = _make_fake_pool_factory(20)

    ret1 = rpe.main(
        argv=common_argv,
        train_eval=counting_train_eval_1,
        strategy_factory=_fake_strategy_factory,
        pool_factory=_pool_factory,
    )
    assert ret1 == 0
    first_count = call_count_first[0]
    assert first_count == 4, f"First run should train 4 cells, got {first_count}"

    ret2 = rpe.main(
        argv=common_argv,
        train_eval=counting_train_eval_2,
        strategy_factory=_fake_strategy_factory,
        pool_factory=_pool_factory,
    )
    assert ret2 == 0
    assert call_count_second[0] == 0, (
        f"Resume run should skip all cells; train_eval called {call_count_second[0]} times"
    )

    df = pd.read_csv(out_dir / "curve.csv")
    assert len(df) == 4, f"curve.csv should still have 4 rows, got {len(df)}"


def test_incremental_csv_written_per_cell(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, tmp_path, n_images=20)

    import run_paper_experiments as rpe  # noqa: PLC0415

    out_dir = tmp_path / "out_incremental"
    strategies = ["random", "uncertainty"]
    budgets = [0.2, 0.5]
    seeds = [0]

    csv_row_counts: list[int] = []

    def snapshot_train_eval(selected_imgs, out_dir_arg, epochs, imgsz, seed, batch):
        result = _make_fake_train_eval()(selected_imgs, out_dir_arg, epochs, imgsz, seed, batch)
        csv_path = out_dir / "curve.csv"
        if csv_path.is_file():
            csv_row_counts.append(len(pd.read_csv(csv_path)))
        return result

    ret = rpe.main(
        argv=[
            "--strategies", *strategies,
            "--budgets", *[str(b) for b in budgets],
            "--seeds", *[str(s) for s in seeds],
            "--out", str(out_dir),
            "--epochs", "1",
        ],
        train_eval=snapshot_train_eval,
        strategy_factory=_fake_strategy_factory,
        pool_factory=_make_fake_pool_factory(20),
    )
    assert ret == 0

    df_final = pd.read_csv(out_dir / "curve.csv")
    assert len(df_final) == 4, f"Expected 4 rows, got {len(df_final)}"

    assert len(csv_row_counts) >= 3, "Should have captured at least 3 mid-run snapshots"
    for k, count in enumerate(csv_row_counts):
        assert count >= k + 1, (
            f"After cell {k+1} completed, curve.csv had only {count} rows "
            f"(expected >= {k+1}); incremental write is not happening per cell"
        )

    df_partial = df_final.head(2)
    df_partial.to_csv(out_dir / "curve.csv", index=False)

    restart_call_count: list[int] = [0]

    def counting_restart_train_eval(selected_imgs, out_dir_arg, epochs, imgsz, seed, batch):
        restart_call_count[0] += 1
        return _make_fake_train_eval()(selected_imgs, out_dir_arg, epochs, imgsz, seed, batch)

    ret2 = rpe.main(
        argv=[
            "--strategies", *strategies,
            "--budgets", *[str(b) for b in budgets],
            "--seeds", *[str(s) for s in seeds],
            "--out", str(out_dir),
            "--epochs", "1",
        ],
        train_eval=counting_restart_train_eval,
        strategy_factory=_fake_strategy_factory,
        pool_factory=_make_fake_pool_factory(20),
    )
    assert ret2 == 0
    assert restart_call_count[0] == 2, (
        f"Restart should train only the 2 remaining cells; got {restart_call_count[0]} calls"
    )
    df_restarted = pd.read_csv(out_dir / "curve.csv")
    assert len(df_restarted) == 4, (
        f"After restart, curve.csv should have 4 rows (no duplicates), got {len(df_restarted)}"
    )


def test_smoke_flag_produces_minimal_output(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, tmp_path, n_images=20)

    import run_paper_experiments as rpe  # noqa: PLC0415

    out_dir = tmp_path / "out_smoke"
    ret = rpe.main(
        argv=["--smoke", "--out", str(out_dir)],
        train_eval=_make_fake_train_eval(),
        strategy_factory=_fake_strategy_factory,
        pool_factory=_make_fake_pool_factory(20),
    )

    assert ret == 0
    df = pd.read_csv(out_dir / "curve.csv")
    assert len(df) == 4, f"Smoke run should produce 4 rows, got {len(df)}"
    assert set(df["strategy"].unique()) == {"random", "rehabal"}
    assert set(df["budget"].round(3).unique()) == {0.05, 0.10}


def test_resume_restores_labeled_idx_correctly(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, tmp_path, n_images=30)

    import run_paper_experiments as rpe  # noqa: PLC0415

    out_dir = tmp_path / "out_labeled_idx"
    strategies = ["random"]
    budgets = [0.1, 0.5, 1.0]
    seeds = [0]

    fresh_calls: list[list[str]] = []

    def recording_train_eval(selected_imgs, out_dir_arg, epochs, imgsz, seed, batch):
        fresh_calls.append(list(selected_imgs))
        return _make_fake_train_eval()(selected_imgs, out_dir_arg, epochs, imgsz, seed, batch)

    ret_fresh = rpe.main(
        argv=[
            "--strategies", *strategies,
            "--budgets", *[str(b) for b in budgets],
            "--seeds", *[str(s) for s in seeds],
            "--out", str(out_dir / "fresh"),
            "--epochs", "1",
        ],
        train_eval=recording_train_eval,
        strategy_factory=_fake_strategy_factory,
        pool_factory=_make_fake_pool_factory(30),
    )
    assert ret_fresh == 0
    assert len(fresh_calls) == 3, f"Expected 3 train_eval calls in fresh run, got {len(fresh_calls)}"

    for b in budgets:
        idx_path = out_dir / "fresh" / f"random_b{b:g}_s0" / "labeled_idx.json"
        assert idx_path.is_file(), f"labeled_idx.json missing for budget {b}"

    import shutil  # noqa: PLC0415
    import json as _json  # noqa: PLC0415

    resume_dir = out_dir / "resume"
    resume_dir.mkdir(parents=True, exist_ok=True)

    b0 = budgets[0]
    cell0_src = out_dir / "fresh" / f"random_b{b0:g}_s0"
    cell0_dst = resume_dir / f"random_b{b0:g}_s0"
    shutil.copytree(str(cell0_src), str(cell0_dst))

    fresh_csv = pd.read_csv(out_dir / "fresh" / "curve.csv")
    round0_row = fresh_csv[np.isclose(fresh_csv["budget"].astype(float), b0)]
    round0_row.to_csv(resume_dir / "curve.csv", index=False)

    resume_calls: list[list[str]] = []

    def resume_recording_train_eval(selected_imgs, out_dir_arg, epochs, imgsz, seed, batch):
        resume_calls.append(list(selected_imgs))
        return _make_fake_train_eval()(selected_imgs, out_dir_arg, epochs, imgsz, seed, batch)

    ret_resume = rpe.main(
        argv=[
            "--strategies", *strategies,
            "--budgets", *[str(b) for b in budgets],
            "--seeds", *[str(s) for s in seeds],
            "--out", str(resume_dir),
            "--epochs", "1",
        ],
        train_eval=resume_recording_train_eval,
        strategy_factory=_fake_strategy_factory,
        pool_factory=_make_fake_pool_factory(30),
    )
    assert ret_resume == 0

    assert len(resume_calls) == 2, (
        f"Resume should train rounds 1+2 only; got {len(resume_calls)} calls"
    )

    assert sorted(resume_calls[0]) == sorted(fresh_calls[1]), (
        "Round 1 labeled imgs differ between fresh and resume run\n"
        f"  fresh[1]={sorted(fresh_calls[1])[:5]}...\n"
        f"  resume[0]={sorted(resume_calls[0])[:5]}..."
    )
    assert sorted(resume_calls[1]) == sorted(fresh_calls[2]), (
        "Round 2 labeled imgs differ between fresh and resume run\n"
        f"  fresh[2]={sorted(fresh_calls[2])[:5]}...\n"
        f"  resume[1]={sorted(resume_calls[1])[:5]}..."
    )
