import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import run_strengthened_baselines as policy

SOURCE = Path(__file__).resolve().parents[2] / "original_code" / "RehabAL-Pose_Code"
import os
SOURCE = Path(os.environ.get("REHAB_TEST_SOURCE", str(SOURCE)))
if not (SOURCE / "src").exists():
    SOURCE = Path("rehabal_exp/original_code/RehabAL-Pose_Code")
sys.path[:0] = [str(SOURCE / "scripts"), str(SOURCE / "src")]


def keypoints(conf=.8):
    return np.column_stack((np.arange(26) + 20., np.arange(26) * 2 + 30., np.full(26, conf)))


class PolicyTests(unittest.TestCase):
    def test_unflip_is_involution_and_swaps_anatomy(self):
        kp = keypoints()
        mirrored = policy.unflip_keypoints(kp, 200)
        self.assertEqual(mirrored[7, 0], 199 - kp[12, 0])
        np.testing.assert_array_equal(policy.unflip_keypoints(mirrored, 200), kp)

    def test_jointmask_explicit_visibility_and_invalid_coordinates(self):
        kp = keypoints()
        kp[0, 2] = .2
        kp[1, 0] = np.nan
        kp[2, 1] = -5
        label, stats = policy.make_pseudo_label(kp, 200, 200, .6)
        np.testing.assert_array_equal(label[:3], np.zeros((3, 3)))
        self.assertEqual(stats["masked_joints"], 3)
        self.assertEqual(set(label[:, 2]), {0., 1.})

    def test_consensus_masks_disagreement_and_averages_good_joints(self):
        kp = keypoints()
        restored = kp.copy()
        restored[0, 0] += 100
        restored[1, 0] += 2
        label, stats = policy.make_pseudo_label(kp, 200, 200, .5, restored, .05)
        self.assertEqual(label[0, 2], 0)
        self.assertEqual(label[1, 0], kp[1, 0] + 1)
        self.assertEqual(stats["consensus_joints"], 25)

    def test_no_detection_and_degenerate_box_do_not_pass(self):
        label, _ = policy.make_pseudo_label(keypoints(), 200, 200, .5, np.zeros((26, 3)))
        self.assertFalse(policy.label_is_usable(label, 6))
        same = np.tile([20., 20., 1.], (26, 1))
        self.assertFalse(policy.label_is_usable(same, 6))

    def test_quantile_is_pool_only_and_missing_has_no_fallback(self):
        kp = keypoints()
        kp[:, 2] = np.linspace(.01, .1, 26)
        self.assertLess(policy.confidence_quantile(kp[None], .25), .04)
        with self.assertRaises(ValueError):
            policy.confidence_quantile(np.zeros((3, 26, 3)), .25)

    def test_gt_guard_blocks_before_reading(self):
        called = []
        def lookup(s):
            called.append(s)
            return keypoints()
        policy.guard_gt(0, lookup, "init")
        with self.assertRaises(RuntimeError):
            policy.guard_gt(1, lookup, "leak")
        self.assertEqual(called, ["init"])

    def test_initial_state_must_match_order_and_coordinates(self):
        labels = {"a": keypoints().tolist(), "b": keypoints(.7).tolist()}
        state = {"labeled": ["a", "b", "later"], "labels": labels}
        policy.validate_initial_state(state, ["a", "b"], labels)
        with self.assertRaises(ValueError):
            policy.validate_initial_state(state, ["b", "a"], labels)
        different = {**labels, "b": keypoints(.9).tolist()}
        with self.assertRaises(ValueError):
            policy.validate_initial_state(state, ["a", "b"], different)

    def test_no_embedding_temporal_parity(self):
        import al_budget_curve as abc
        from rehabal_pose.al import get_strategy
        images = [f"clip_rep00_f{i:04d}.jpg" for i in range(8)]
        class FakeModel:
            def __init__(self):
                self.embedding_calls = 0
            def predict(self, paths):
                return [SimpleNamespace(sample_id=Path(p).stem,
                        keypoints=keypoints() + [int(Path(p).stem[-4:]) ** 2, 0, 0]) for p in paths]
            def extract_embeddings(self, paths):
                self.embedding_calls += 1
                return np.ones((len(paths), 4))
        model = FakeModel()
        original = abc._build_pool(images, model, {})
        calls = model.embedding_calls
        changed = policy.build_pool_without_embeddings(abc._build_pool, images, model, {})
        self.assertEqual(model.embedding_calls, calls)
        self.assertIsNone(changed.embeddings)
        np.testing.assert_array_equal(original.predictions, changed.predictions)
        np.testing.assert_array_equal(get_strategy("temporal").score(original),
                                      get_strategy("temporal").score(changed))

    def test_original_loop_adapter_keeps_masks_and_never_counts_correction(self):
        import run_clinician_experiment as runner
        import al_budget_curve as abc
        from rehabal_pose.review.clinician_loop import ClinicianLoop, SessionState, Decision
        from rehabal_pose.al.base import Pool
        from PIL import Image
        args = SimpleNamespace(min_joints=6, confidence_quantile=.25, flip_tolerance=.05,
                               inference_batch=4, budgets=[.25, .5])
        context = SimpleNamespace(train_seconds=0.)
        originals = {k: getattr(runner, k) for k in
                     ("_build_pool", "_source_identity", "CONDITIONS", "_decision_fn_for", "ClinicianLoop")}
        try:
            adapted = policy.install_runtime(args, runner, abc, ClinicianLoop, context)
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                ids = [f"clip_rep00_f{i:04d}" for i in range(8)]
                for sid in ids:
                    Image.new("RGB", (200, 200)).save(root / f"{sid}.jpg")
                predictions = np.stack([keypoints() for _ in ids])
                predictions[:, :5, 2] = .01
                pool = Pool(ids, predictions=predictions, clip_ids=["clip"]*8, frame_ids=list(range(8)))
                stored = []
                def train(stems, labels, out_dir, seed):
                    stored.append(json.loads(json.dumps(labels)))
                    weight = Path(out_dir) / "best.pt"
                    weight.write_bytes(b"mock-only-not-a-model")
                    context.train_seconds = .1
                    return {"weights": str(weight), "map50": .5, "map50_95": .3}
                state = SessionState(session_id="A_jointmask_s0", condition="A_jointmask", seed=0,
                                     strategy="temporal", budget_per_round=2, round=0, labeled=[], labels={}, history=[], status="active")
                loop = adapted(state, infer_fn=None, score_fn=lambda w: (pool, {"score": np.arange(8.)}),
                               train_eval_fn=train, gt_lookup=lambda s: None, output_root=root,
                               image_dir=root, run_mode="auto", train_image_stems=set(ids))
                loop.apply_decisions([Decision(sample_id=s, action="correct", keypoints=keypoints().tolist())
                                      for s in ids[:2]])
                loop.commit_and_train()
                loop.state.budget_per_round = 2
                record = loop.run_auto_round(runner._decision_fn_for("A_jointmask", None, None))
                self.assertEqual(record.n_corrected, 0)
                self.assertEqual(record.clinician, "none")
                self.assertEqual(record.n_labeled, 4)
                new_ids = set(stored[-1]) - set(ids[:2])
                for sid in new_ids:
                    self.assertEqual(np.asarray(stored[-1][sid])[:5, 2].sum(), 0)
                audits = [json.loads(line) for line in
                          (root / "A_jointmask_s0" / "audit_rounds.jsonl").read_text().splitlines()]
                self.assertEqual(audits[-1]["gt_additional_frames"], 0)
                self.assertEqual(audits[-1]["automatic_masked_joints"], 10)
                self.assertGreater(audits[-1]["proposed_frames"], 0)
        finally:
            for key, value in originals.items():
                setattr(runner, key, value)


    def test_full_session_flip_branch_and_completed_reuse(self):
        import run_clinician_experiment as runner
        import al_budget_curve as abc
        from rehabal_pose.review.clinician_loop import ClinicianLoop
        from rehabal_pose.models.yolo_pose import YoloPoseModel
        from rehabal_pose.al.base import Pool
        from PIL import Image
        args = SimpleNamespace(min_joints=6, confidence_quantile=.25, flip_tolerance=.05,
                               inference_batch=3, budgets=[.25, .5])
        context = SimpleNamespace(train_seconds=0.)
        keys = ("_build_pool", "_source_identity", "CONDITIONS", "_decision_fn_for", "ClinicianLoop", "IMAGES_TRAIN")
        originals = {k: getattr(runner, k) for k in keys}
        try:
            policy.install_runtime(args, runner, abc, ClinicianLoop, context)
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                images = root / "images"
                images.mkdir()
                runner.IMAGES_TRAIN = images
                ids = [f"clip_rep00_f{i:04d}" for i in range(8)]
                for sid in ids:
                    Image.new("RGB", (200, 200)).save(images / f"{sid}.jpg")
                kp = keypoints()
                kp[:5, 2] = .01
                pool = Pool(ids, predictions=np.stack([kp] * len(ids)),
                            clip_ids=["clip"]*8, frame_ids=list(range(8)))
                calls, gt_calls = [], []
                def train(stems, labels, out_dir, seed):
                    calls.append(list(stems))
                    weight = Path(out_dir) / "best.pt"
                    weight.write_bytes(b"mock-only-not-a-model")
                    context.train_seconds = .1
                    return {"weights": str(weight), "map50": .5, "map50_95": .3}
                def gt(stem):
                    gt_calls.append(stem)
                    return policy.guard_gt(context.loop.state.round, lambda s: keypoints(), stem)
                mirrored = policy.unflip_keypoints(kp, 200)
                class FakeNet:
                    def predict(self, source, **kwargs):
                        return [SimpleNamespace(keypoints=SimpleNamespace(data=mirrored[None]),
                                boxes=SimpleNamespace(conf=np.array([.9]))) for _ in source]
                callables = (None, lambda w: (pool, {"score": np.arange(8.)}), train, gt)
                with patch.dict(sys.modules, {"cv2": SimpleNamespace(imread=lambda p: np.asarray(Image.open(p)))}), \
                     patch.object(YoloPoseModel, "load", lambda self, weights: self), \
                     patch.object(YoloPoseModel, "_ensure_model", return_value=FakeNet()), \
                     patch.object(YoloPoseModel, "_resolve_device", return_value="cpu"):
                    rows = runner._run_session("A_flipcons", 0, 0., args.budgets, ids, len(ids),
                                               callables, None, root / "out", "temporal",
                                               run_config={"mock": True}, input_identity={"mock": True})
                    self.assertEqual(len(rows), 2)
                    self.assertEqual(rows[1]["n_corrected"], 0)
                    self.assertEqual(len(gt_calls), 2)
                    self.assertEqual(len(calls), 2)
                    again = runner._run_session("A_flipcons", 0, 0., args.budgets, ids, len(ids),
                                                callables, None, root / "out", "temporal",
                                                run_config={"mock": True}, input_identity={"mock": True})
                    self.assertEqual(len(again), 2)
                    self.assertEqual(len(calls), 2)
                audit_path = root / "out/A_flipcons_s0/audit_rounds.jsonl"
                audit = [json.loads(line) for line in audit_path.read_text().splitlines()]
                self.assertEqual(audit[-1]["consensus_joints"], 42)
                self.assertGreater(audit[-1]["extra_flip_inference_images"], 0)
                self.assertTrue((root / "out/A_flipcons_s0/completion.json").exists())
        finally:
            for key, value in originals.items():
                setattr(runner, key, value)

if __name__ == "__main__":
    unittest.main()
