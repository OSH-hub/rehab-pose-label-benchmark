"""Tests for the YOLO-Pose training pipeline (``rehabal_pose.pipelines.train_pose``).

Mock-first: the default suite never imports torch/ultralytics/cv2, never touches
a GPU, never downloads weights, and never trains. ``YoloPoseModel.train`` is
monkeypatched to a fake so we exercise argparse wiring + config loading +
``metrics.json`` writing in isolation.

The single real-training test is marked ``@pytest.mark.slow`` AND env-gated on
``REHABAL_RUN_SLOW`` (plus ``pytest.importorskip("ultralytics")``), so it is
skipped by default.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from rehabal_pose.pipelines.train_pose import (
    load_model_config,
    main,
    train_pose,
)

HEAVY = ("torch", "ultralytics", "cv2")

_REPO = Path(__file__).resolve().parent.parent
_MODEL_CONFIG = _REPO / "configs" / "models" / "yolo_pose.yaml"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def _fake_result(save_dir: Path, weights: Path) -> dict:
    return {
        "weights": str(weights),
        "last_weights": None,
        "save_dir": str(save_dir),
        "map50_95": 0.5,
        "map50": 0.7,
        "metrics": {"metrics/mAP50-95(P)": 0.5},
        "epochs": 1,
        "device": "cpu",
    }


@pytest.fixture()
def patched_train(monkeypatch, tmp_path):
    """Monkeypatch YoloPoseModel.train -> records call + returns a fake dict."""
    from rehabal_pose.models.yolo_pose import YoloPoseModel

    captured: dict = {}
    save_dir = tmp_path / "run"
    weights = save_dir / "weights" / "best.pt"

    def fake_train(self, data_yaml, output_dir=None, config=None):
        captured["data_yaml"] = str(data_yaml)
        captured["output_dir"] = None if output_dir is None else str(output_dir)
        captured["config"] = dict(self.config)
        captured["call_config"] = dict(config) if config is not None else None
        return _fake_result(save_dir, weights)

    monkeypatch.setattr(YoloPoseModel, "train", fake_train)
    return captured


# ---------------------------------------------------------------------------
# no-heavy-import guarantee
# ---------------------------------------------------------------------------


def test_importing_pipeline_does_not_import_heavy_deps():
    for name in list(sys.modules):
        if name in HEAVY or name.startswith(tuple(h + "." for h in HEAVY)):
            del sys.modules[name]
    importlib.import_module("rehabal_pose.pipelines.train_pose")
    for h in HEAVY:
        assert h not in sys.modules, f"{h} imported at module load time"


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------


def test_load_model_config_base():
    cfg = load_model_config(_MODEL_CONFIG)
    assert cfg["weights"] == "yolo11n-pose.pt"
    assert cfg["num_keypoints"] == 26
    assert cfg["imgsz"] == 640
    assert cfg["epochs"] == 50
    assert cfg["device"] == "auto"
    assert "patience" in cfg


def test_load_model_config_smoke_profile_overlays_base():
    cfg = load_model_config(_MODEL_CONFIG, profile="smoke")
    # overlaid keys
    assert cfg["epochs"] == 1
    assert cfg["imgsz"] == 320
    assert cfg["batch"] == 2
    # non-overlaid base keys survive
    assert cfg["weights"] == "yolo11n-pose.pt"
    assert cfg["num_keypoints"] == 26


def test_load_model_config_none_returns_empty():
    assert load_model_config(None) == {}


def test_load_model_config_unknown_profile_raises():
    with pytest.raises(KeyError):
        load_model_config(_MODEL_CONFIG, profile="does-not-exist")


# ---------------------------------------------------------------------------
# programmatic train_pose (fake backend)
# ---------------------------------------------------------------------------


def test_train_pose_constructs_model_and_returns_metrics(patched_train, tmp_path):
    out = tmp_path / "out"
    result = train_pose("data.yaml", str(out), {"epochs": 3, "imgsz": 320})
    assert result["map50_95"] == 0.5
    # config reached YoloPoseModel.__init__ (self.config) and the train() call.
    assert patched_train["config"]["epochs"] == 3
    assert patched_train["config"]["imgsz"] == 320
    assert patched_train["data_yaml"] == "data.yaml"
    assert patched_train["output_dir"] == str(out)


# ---------------------------------------------------------------------------
# CLI main (fake backend)
# ---------------------------------------------------------------------------


def test_main_writes_metrics_json_to_output_dir(patched_train, tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("kpt_shape: [26, 3]\n", encoding="utf-8")
    out = tmp_path / "outdir"

    rc = main(
        [
            "--data", str(data_yaml),
            "--output-dir", str(out),
            "--epochs", "3",
            "--imgsz", "320",
            "--batch", "2",
            "--device", "cpu",
            "--seed", "7",
        ]
    )
    assert rc == 0
    metrics_file = out / "metrics.json"
    assert metrics_file.exists()
    saved = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert saved["map50_95"] == 0.5
    assert saved["weights"].endswith("best.pt")

    # CLI overrides propagated into the model config.
    cfg = patched_train["config"]
    assert cfg["epochs"] == 3
    assert cfg["imgsz"] == 320
    assert cfg["batch"] == 2
    assert cfg["device"] == "cpu"
    assert cfg["seed"] == 7
    assert patched_train["data_yaml"] == str(data_yaml)


def test_main_with_model_config_and_smoke_profile(patched_train, tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("x\n", encoding="utf-8")
    out = tmp_path / "outdir2"

    rc = main(
        [
            "--data", str(data_yaml),
            "--model-config", str(_MODEL_CONFIG),
            "--profile", "smoke",
            "--output-dir", str(out),
        ]
    )
    assert rc == 0
    assert (out / "metrics.json").exists()
    cfg = patched_train["config"]
    # smoke profile values flowed through (no CLI override for these).
    assert cfg["epochs"] == 1
    assert cfg["imgsz"] == 320
    assert cfg["batch"] == 2
    assert cfg["weights"] == "yolo11n-pose.pt"


def test_main_falls_back_to_save_dir_when_no_output_dir(patched_train, tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("x\n", encoding="utf-8")
    rc = main(["--data", str(data_yaml)])
    assert rc == 0
    # The fake's save_dir is tmp_path/'run'; metrics.json lands there.
    assert (tmp_path / "run" / "metrics.json").exists()


def test_main_requires_data():
    with pytest.raises(SystemExit):
        main([])


# ---------------------------------------------------------------------------
# SLOW: real 1-epoch smoke training on a tiny synthetic dataset.
# Skipped by default; run with REHABAL_RUN_SLOW=1 on a machine with ultralytics.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_training_one_epoch(tmp_path):
    if not os.environ.get("REHABAL_RUN_SLOW"):
        pytest.skip("set REHABAL_RUN_SLOW=1 to run the real smoke-training test")
    pytest.importorskip("ultralytics")
    pytest.importorskip("torch")
    PIL = pytest.importorskip("PIL.Image")

    from rehabal_pose.data.yolo_dataset import (
        REHAB26_NUM_KEYPOINTS,
        write_pose_data_yaml,
        yolo_label_line,
    )
    from rehabal_pose.models.yolo_pose import is_available

    if not is_available():
        pytest.skip("ultralytics/torch not importable")

    rng = np.random.default_rng(0)
    ds = tmp_path / "ds"
    for split, n in (("train", 4), ("val", 2)):
        img_dir = ds / "images" / split
        lbl_dir = ds / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (rng.uniform(0, 255, size=(320, 320, 3))).astype("uint8")
            PIL.fromarray(arr).save(img_dir / f"frame_{i:03d}.jpg")
            kp = np.concatenate(
                [
                    rng.uniform(0.3, 0.7, size=(REHAB26_NUM_KEYPOINTS, 2)),
                    np.full((REHAB26_NUM_KEYPOINTS, 1), 2.0),
                ],
                axis=1,
            )
            line = yolo_label_line([0.5, 0.5, 0.5, 0.5], kp)
            (lbl_dir / f"frame_{i:03d}.txt").write_text(line + "\n", encoding="utf-8")

    data_yaml = write_pose_data_yaml(
        ds, ds / "images" / "train", ds / "images" / "val"
    )

    cfg = load_model_config(_MODEL_CONFIG, profile="smoke")
    cfg["device"] = "cpu"  # keep deterministic / CI-friendly
    result = train_pose(str(data_yaml), str(tmp_path / "run"), cfg)

    assert result["save_dir"]
    assert Path(result["save_dir"]).exists()
    assert result["epochs"] == 1
