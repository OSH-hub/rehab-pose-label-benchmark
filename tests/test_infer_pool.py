from __future__ import annotations

import sys

import numpy as np
import pytest

from rehabal_pose.al import Pool
from rehabal_pose.models.pose_base import Prediction
from rehabal_pose.pipelines import infer_pool as ip


K = 26
D = 8


class FakeModel:
    num_keypoints = K

    def __init__(self, *, miss: set[str] | None = None, seed: int = 0):
        self.miss = miss or set()
        self.rng = np.random.default_rng(seed)
        self.predict_calls: list[list[str]] = []
        self.embed_calls: list[list[str]] = []

    def predict(self, inputs, config=None):
        from pathlib import Path

        paths = [inputs] if isinstance(inputs, str) else list(inputs)
        self.predict_calls.append([str(p) for p in paths])
        out: list[Prediction] = []
        for idx, p in enumerate(paths):
            stem = Path(str(p)).stem
            if stem in self.miss:
                continue
            kpts = self.rng.random((K, 3)).astype(np.float64)
            kpts[:, 2] = np.clip(kpts[:, 2], 0.0, 1.0)
            out.append(
                Prediction(
                    sample_id=stem,
                    keypoints=kpts,
                    bbox=[1.0, 2.0, 3.0, 4.0],
                    score=0.5 + 0.01 * idx,
                )
            )
        return out

    def extract_embeddings(self, inputs, config=None):
        paths = [inputs] if isinstance(inputs, str) else list(inputs)
        self.embed_calls.append([str(p) for p in paths])
        return self.rng.random((len(paths), D)).astype(np.float64)


def _items():
    return [
        {
            "image_path": "frames/PM_000_c17_000010.jpg",
            "sample_id": "PM_000_c17_000010",
            "video_id": "PM_000",
            "frame_id": 10,
            "disease": "stroke",
            "exercise": "Ex1",
            "camera_id": "c17",
            "standardtype": "correct",
        },
        {
            "image_path": "frames/PM_000_c17_000005.jpg",
            "sample_id": "PM_000_c17_000005",
            "video_id": "PM_000",
            "frame_id": 5,
            "disease": "stroke",
            "exercise": "Ex1",
            "camera_id": "c17",
            "standardtype": "correct",
        },
        {
            "image_path": "frames/PM_001_c18_000020.jpg",
            "sample_id": "PM_001_c18_000020",
            "video_id": "PM_001",
            "frame_id": 20,
            "disease": "parkinson",
            "exercise": "Ex2",
            "camera_id": "c18",
            "standardtype": "incorrect",
        },
    ]


def test_no_heavy_imports_on_load():
    import importlib

    importlib.import_module("rehabal_pose.pipelines.infer_pool")
    for heavy in ("torch", "ultralytics", "cv2"):
        assert heavy not in sys.modules, f"{heavy} leaked into sys.modules"


def test_infer_pool_builds_valid_pool():
    items = _items()
    model = FakeModel()
    pool = ip.infer_pool(model, items)

    assert isinstance(pool, Pool)
    assert pool.size == len(items)
    assert pool.sample_ids == [r["sample_id"] for r in items]

    assert pool.predictions is not None
    assert pool.predictions.shape == (len(items), K, 3)

    assert pool.clip_ids == ["PM_000", "PM_000", "PM_001"]
    assert pool.frame_ids == [10, 5, 20]
    groups = pool.clip_groups()
    assert set(groups) == {"PM_000", "PM_001"}
    assert groups["PM_000"] == [1, 0]

    assert pool.meta("disease") == ["stroke", "stroke", "parkinson"]
    assert pool.meta("exercise") == ["Ex1", "Ex1", "Ex2"]
    assert pool.meta("camera_id") == ["c17", "c17", "c18"]

    assert pool.embeddings is None
    assert len(model.predict_calls) == 1
    assert model.predict_calls[0] == [r["image_path"] for r in items]


def test_infer_pool_with_embeddings_aligned():
    items = _items()
    model = FakeModel()
    pool = ip.infer_pool(model, items, config={"with_embeddings": True})

    assert pool.embeddings is not None
    assert pool.embeddings.shape == (len(items), D)
    assert len(model.embed_calls) == 1
    assert model.embed_calls[0] == [r["image_path"] for r in items]


def test_missing_detection_yields_zero_row():
    items = _items()
    miss = {"PM_001_c18_000020"}
    model = FakeModel(miss=miss)
    pool = ip.infer_pool(model, items)

    assert pool.size == len(items)
    assert np.allclose(pool.predictions[2], 0.0)
    assert pool.confidence()[2].sum() == 0.0
    assert pool.confidence()[0].sum() > 0.0
    det = pool.meta("det_score")
    assert det is not None
    assert det[2] is None
    assert det[0] is not None


def test_accepts_plain_path_list():
    paths = ["a/x1.jpg", "a/x2.jpg"]
    model = FakeModel()
    pool = ip.infer_pool(model, paths)
    assert pool.sample_ids == ["x1", "x2"]
    assert pool.clip_ids is None
    assert pool.predictions.shape == (2, K, 3)


@pytest.mark.parametrize("ext", [".npz", ".parquet"])
def test_save_and_load_round_trip(tmp_path, ext):
    items = _items()
    model = FakeModel()
    out = tmp_path / f"pool{ext}"
    pool = ip.infer_pool(model, items, output_path=out, config={"with_embeddings": True})
    assert out.is_file()

    loaded = ip.load_pool(out)
    assert loaded.sample_ids == pool.sample_ids
    np.testing.assert_allclose(loaded.predictions, pool.predictions)
    assert loaded.embeddings is not None
    np.testing.assert_allclose(loaded.embeddings, pool.embeddings)
    assert list(loaded.clip_ids) == list(pool.clip_ids)
    assert list(loaded.frame_ids) == list(pool.frame_ids)
    assert loaded.meta("disease") == pool.meta("disease")
    assert loaded.meta("exercise") == pool.meta("exercise")


def test_output_path_persists_via_infer_pool(tmp_path):
    items = _items()
    model = FakeModel()
    out = tmp_path / "pool.npz"
    ip.infer_pool(model, items, output_path=out)
    assert out.is_file()
    loaded = ip.load_pool(out)
    assert loaded.size == len(items)


def test_items_from_manifest_with_images_dir():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "sample_id": ["PM_000_c17_000010", "PM_001_c18_000020"],
            "video_id": ["PM_000", "PM_001"],
            "frame_id": [10, 20],
            "video_path": ["videos/PM_000.mp4", "videos/PM_001.mp4"],
            "disease": ["stroke", "parkinson"],
            "exercise": ["Ex1", "Ex2"],
            "camera_id": ["c17", "c18"],
        }
    )
    items = ip.items_from_manifest(df, images_dir="frames")
    assert items[0]["image_path"].replace("\\", "/") == "frames/PM_000_c17_000010.jpg"
    assert items[0]["sample_id"] == "PM_000_c17_000010"
    assert items[0]["video_id"] == "PM_000"

    model = FakeModel()
    pool = ip.infer_pool(model, df, config={"num_keypoints": K})
    assert pool.size == 2
    assert pool.clip_ids == ["PM_000", "PM_001"]
    assert pool.meta("disease") == ["stroke", "parkinson"]


def test_bad_item_type_raises():
    with pytest.raises(TypeError):
        ip.infer_pool(FakeModel(), [123])
    with pytest.raises(ValueError):
        ip.infer_pool(FakeModel(), [{"sample_id": "x"}])
