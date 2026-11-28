import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_clinician_experiment as runner
from rehabal_pose.review.clinician_loop import (
    ClinicianLoop, Decision, RoundTrainingError, SessionState,
)


@pytest.mark.parametrize("arguments", [
    ["--conditions", "A", "A"], ["--seeds", "7", "7"],
    ["--sigmas", "0", "0.0"], ["--sigmas", "nan"], ["--sigmas", "-1"],
])
def test_invalid_grid_is_rejected_before_loading_data(arguments, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid grid must not load data or publish results")
    monkeypatch.setattr(runner.glob, "glob", unexpected)
    with pytest.raises(SystemExit) as exc:
        runner.main(arguments)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    images = dataset / "images" / "train"
    images.mkdir(parents=True)
    monkeypatch.setattr(runner, "DATASET", dataset)
    monkeypatch.setattr(runner, "IMAGES_TRAIN", images)
    monkeypatch.setattr(runner, "MANIFEST", tmp_path / "manifest.parquet")
    monkeypatch.setattr(runner, "SEG_CSV", tmp_path / "seg.csv")
    stems = [f"fixture_rep00_f{i}" for i in range(10)]
    kp = [[100.0, 200.0, 2.0]] * 26
    calls = []
    behavior = {"fail_on": None, "invalid": None}

    def train(labeled, labels, out_dir, seed):
        calls.append(list(labeled))
        if len(calls) == behavior["fail_on"]:
            if behavior["invalid"] is not None:
                return behavior["invalid"]
            raise RuntimeError("synthetic late training failure")
        (Path(out_dir) / "mock-best.pt").write_bytes(b"synthetic weights")
        return {"map50": 0.8, "map50_95": 0.5,
                "weights": str(Path(out_dir) / "mock-best.pt")}

    callables = (
        lambda weights, selected: {s: kp for s in selected},
        lambda weights: (SimpleNamespace(sample_ids=stems),
                         {"score": np.arange(10, dtype=float)}),
        train, lambda stem: np.asarray(kp),
    )
    config = {"epochs": 2, "epochs_round0": 3, "imgsz": 320,
              "batch": 2, "weights": "mock.pt"}

    def run(**overrides):
        kwargs = dict(condition="A", seed=0, sigma=0.0,
                      budgets=[0.2, 0.4, 0.6, 0.8], pool_stems=stems, N=10,
                      callables=callables, seg_table=None, out_root=tmp_path / "runs",
                      strategy_name="temporal", run_config=config)
        kwargs.update(overrides)
        return runner._run_session(**kwargs)

    return SimpleNamespace(run=run, calls=calls, behavior=behavior,
                           config=config, callables=callables, stems=stems,
                           root=tmp_path / "runs", session=tmp_path / "runs" / "A_s0",
                           dataset=dataset, images=images)


def test_clean_completion_is_reused_without_training(experiment):
    first = experiment.run()
    assert [len(x) for x in experiment.calls] == [2, 4, 6, 8]
    assert experiment.run() == first
    assert len(experiment.calls) == 4
    assert [r["budget"] for r in first] == [0.2, 0.4, 0.6, 0.8]
    assert [r["n_labeled"] for r in first] == [2, 4, 6, 8]
    assert len(pd.read_csv(experiment.session / "curve.csv")) == 4
    assert json.loads((experiment.session / "state.json").read_text())["status"] == "done"


@pytest.mark.parametrize("damage", ["delete", "modify"])
def test_every_budget_checkpoint_is_required_for_reuse(experiment, damage):
    experiment.run()
    checkpoint = experiment.session / "round_0" / "mock-best.pt"
    if damage == "delete":
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(b"changed intermediate weights")
    with pytest.raises(runner.SessionReuseError, match="Round 0 checkpoint"):
        experiment.run()
    assert len(experiment.calls) == 4


@pytest.mark.parametrize("failure_stage", ["curve", "state", "persistent_disk"])
def test_storage_failure_locks_same_loop_without_granting_retry_budget(tmp_path, monkeypatch, failure_stage):
    state = SessionState("storage", "A", 0, "temporal", 1, 0, [], {}, [], "active")
    calls = []
    def train(labeled, labels, out_dir, seed):
        calls.append(list(labeled))
        weights = Path(out_dir) / "best.pt"
        weights.write_bytes(b"checkpoint")
        return {"map50": .8, "map50_95": .5, "weights": str(weights)}
    loop = ClinicianLoop(state, infer_fn=lambda *a: {}, score_fn=lambda *a: None,
                         train_eval_fn=train, gt_lookup=lambda *a: None,
                         output_root=tmp_path, image_dir=tmp_path / "images")
    kp = [[100., 200., 2.]] * 26
    loop.apply_decisions([Decision("one", "correct", kp)])
    loop.commit_and_train()
    previous_curve = (tmp_path / "storage" / "curve.csv").read_bytes()
    previous_weights = state.current_weights
    save = loop._save_state
    def fail_curve(rec):
        raise OSError("synthetic disk failure")
    def fail_state():
        if state.status == "training" and state.round == 2:
            raise OSError("synthetic state failure")
        return save()
    if failure_stage == "state":
        monkeypatch.setattr(loop, "_save_state", fail_state)
    else:
        monkeypatch.setattr(loop, "_append_curve_csv", fail_curve)
        if failure_stage == "persistent_disk":
            def fail_after_training():
                if len(calls) >= 2:
                    raise OSError("persistent disk failure")
                return save()
            monkeypatch.setattr(loop, "_save_state", fail_after_training)
    loop.apply_decisions([Decision("two", "correct", kp)])
    with pytest.raises(RoundTrainingError, match="persistence failed"):
        loop.commit_and_train()
    assert state.status == "failed" and state.round == 1 and len(state.history) == 1
    assert state.labeled == ["one", "two"]
    assert loop._current_weights == state.current_weights == previous_weights
    assert (tmp_path / "storage" / "curve.csv").read_bytes() == previous_curve
    with pytest.raises(RoundTrainingError, match="do not grant another label budget"):
        loop.commit_and_train()
    assert len(calls) == 2
    persisted = ClinicianLoop.load_state(tmp_path / "storage")
    assert persisted.status in {"failed", "training"}


@pytest.mark.parametrize("change", [
    {"strategy_name": "random"}, {"budgets": [0.1, 0.3, 0.5, 0.7]},
    {"conf_threshold": 0.7}, {"verify_oks_threshold": 0.6},
    {"run_config": {"epochs": 9}}, {"run_config": {"weights": "another.pt"}},
])
def test_wrong_configuration_is_never_relabelled_or_restarted(experiment, change):
    experiment.run()
    before = {p.name: p.read_bytes() for p in experiment.session.iterdir() if p.is_file()}
    with pytest.raises(runner.SessionReuseError, match="fingerprint differs"):
        experiment.run(restart_incomplete=True, **change)
    assert len(experiment.calls) == 4
    assert before == {p.name: p.read_bytes() for p in experiment.session.iterdir() if p.is_file()}


@pytest.mark.parametrize("change", [
    ("map50", np.nan), ("map50_95", np.inf), ("map50", -0.01),
    ("round", 2), ("n_labeled", 3), ("n_accepted", 1), ("n_corrected", 9),
])
def test_invalid_cached_curve_is_preserved_and_refused(experiment, change):
    experiment.run()
    path = experiment.session / "curve.csv"
    curve = pd.read_csv(path)
    curve.loc[0, change[0]] = change[1]
    curve.to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(runner.SessionReuseError, match="incomplete/invalid"):
        experiment.run()
    assert path.read_bytes() == before
    assert len(experiment.calls) == 4


def test_legacy_row_count_match_is_not_evidence_of_completion(experiment):
    experiment.session.mkdir(parents=True)
    path = experiment.session / "curve.csv"
    path.write_text("round,n_labeled,map50,map50_95\n0,2,.8,.5\n1,4,.8,.5\n2,6,.8,.5\n3,8,.8,.5\n")
    before = path.read_bytes()
    with pytest.raises(runner.SessionReuseError, match="legacy.*new --out"):
        experiment.run(restart_incomplete=True)
    assert not experiment.calls
    assert path.read_bytes() == before


def test_partial_crash_refuses_append_and_explicit_restart_archives(experiment):
    experiment.behavior["fail_on"] = 2
    with pytest.raises(RoundTrainingError, match="late training"):
        experiment.run()
    curve_path = experiment.session / "curve.csv"
    partial = curve_path.read_bytes()
    assert len(pd.read_csv(curve_path)) == 1
    with pytest.raises(runner.SessionReuseError, match="restart-incomplete"):
        experiment.run()
    assert len(experiment.calls) == 2
    assert curve_path.read_bytes() == partial
    experiment.behavior["fail_on"] = None
    result = experiment.run(restart_incomplete=True)
    assert [r["n_labeled"] for r in result] == [2, 4, 6, 8]
    archived = list((experiment.root / "_archive").glob("*/curve.csv"))
    assert len(archived) == 1 and archived[0].read_bytes() == partial
    assert len(pd.read_csv(curve_path)) == 4
    assert experiment.run() == result
    assert [len(s) for s in experiment.calls] == [2, 4, 2, 4, 6, 8]


def test_interrupted_after_last_curve_row_requires_completion_marker(experiment):
    experiment.run()
    (experiment.session / "completion.json").unlink()
    with pytest.raises(runner.SessionReuseError, match="incomplete/invalid"):
        experiment.run()
    assert len(experiment.calls) == 4


def test_abrupt_interrupt_leaves_one_row_and_never_appends_fresh_state(experiment):
    callables = list(experiment.callables)
    original_train = callables[2]
    def interrupted_train(*args):
        if len(experiment.calls) == 1:
            raise KeyboardInterrupt("synthetic process interruption")
        return original_train(*args)
    callables[2] = interrupted_train
    with pytest.raises(KeyboardInterrupt):
        experiment.run(callables=tuple(callables))
    state = ClinicianLoop.load_state(experiment.session)
    assert state.status == "training" and state.round == 1
    assert len(state.labeled) == 4 and len(state.history) == 1
    before = (experiment.session / "curve.csv").read_bytes()
    assert len(pd.read_csv(experiment.session / "curve.csv")) == 1
    with pytest.raises(runner.SessionReuseError, match="restart-incomplete"):
        experiment.run()
    assert len(experiment.calls) == 1
    assert (experiment.session / "curve.csv").read_bytes() == before


@pytest.mark.parametrize("relative_path", ["images/train/fixture_rep00_f0.jpg",
                                          "labels/train/fixture_rep00_f0.txt",
                                          "images/val/validation.jpg", "labels/val/validation.txt"])
def test_changed_dataset_bytes_invalidate_cache(experiment, relative_path):
    path = experiment.dataset / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"before")
    experiment.run()
    path.write_bytes(b"after!")
    with pytest.raises(runner.SessionReuseError, match="fingerprint differs"):
        experiment.run()
    assert len(experiment.calls) == 4


@pytest.mark.parametrize("bad_result", [
    None, [], "bad", {}, {"map50": .8, "map50_95": .5, "weights": None},
    {"map50": .8, "map50_95": .5, "weights": ""},
    {"map50": .8, "map50_95": .5, "weights": "missing.pt"},
    {"map50": .8, "map50_95": .5, "weights": "empty.pt"},
    {"map50": np.nan, "map50_95": .5, "weights": "new.pt"},
    {"map50": .8, "map50_95": np.inf, "weights": "new.pt"},
    {"map50": .8, "map50_95": None, "weights": "new.pt"},
])
def test_late_invalid_return_records_failure_without_success_round(tmp_path, bad_result):
    state = SessionState("late", "A", 0, "temporal", 1, 0, [], {}, [], "active")
    previous_weights = str(tmp_path / "previous.pt")
    Path(previous_weights).write_bytes(b"previous mock checkpoint")
    if isinstance(bad_result, dict) and bad_result.get("weights") == "new.pt":
        new_weights = tmp_path / "new.pt"
        new_weights.write_bytes(b"new mock checkpoint")
        bad_result = dict(bad_result, weights=str(new_weights))
    if isinstance(bad_result, dict) and bad_result.get("weights") == "empty.pt":
        empty_weights = tmp_path / "empty.pt"
        empty_weights.touch()
        bad_result = dict(bad_result, weights=str(empty_weights))
    returned = [{"map50": .8, "map50_95": .5, "weights": previous_weights}, bad_result]
    loop = ClinicianLoop(state, infer_fn=lambda *a: {}, score_fn=lambda *a: None,
                         train_eval_fn=lambda *a: returned.pop(0), gt_lookup=lambda *a: None,
                         output_root=tmp_path, image_dir=tmp_path / "images")
    kp = [[100., 200., 2.]] * 26
    loop.apply_decisions([Decision("one", "correct", kp)])
    loop.commit_and_train()
    previous_curve = (tmp_path / "late" / "curve.csv").read_bytes()
    loop.apply_decisions([Decision("two", "correct", kp)])
    with pytest.raises(RoundTrainingError):
        loop.commit_and_train()
    assert state.status == "failed" and state.round == 1
    assert len(state.history) == 1 and state.history[0].pose_map50_95 == .5
    assert state.labeled == ["one", "two"]
    assert loop._current_weights == state.current_weights == previous_weights
    assert (tmp_path / "late" / "curve.csv").read_bytes() == previous_curve
    saved = ClinicianLoop.load_state(tmp_path / "late")
    assert saved.status == "failed" and saved.failure["accepted"] == ["two"]
    assert (tmp_path / "late" / "round_1" / "failure.json").is_file()
    with pytest.raises(RoundTrainingError, match="do not grant another label budget"):
        loop.commit_and_train()


def test_late_exception_preserves_previous_result_and_failed_acceptance(experiment):
    experiment.behavior["fail_on"] = 2
    with pytest.raises(RoundTrainingError, match="synthetic late training failure"):
        experiment.run()
    state = ClinicianLoop.load_state(experiment.session)
    assert state.round == 1 and state.status == "failed"
    assert len(state.history) == 1 and len(state.labeled) == 4
    assert state.current_weights.endswith("round_0\\mock-best.pt") or state.current_weights.endswith("round_0/mock-best.pt")
    assert state.failure["error_type"] == "RuntimeError"
    assert not (experiment.session / "completion.json").exists()


def test_main_failure_subprocess_has_nonzero_exit_and_preserves_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "results"
    old_session = out / "A_s0"
    old_session.mkdir(parents=True)
    (old_session / "curve.csv").write_text("saved session\n")
    master = out / "clinician_curve.csv"
    master.write_text("previous valid master\n")
    script = tmp_path / "failure_probe.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\nimport pandas as pd\n"
        f"sys.path.insert(0, {str(root / 'scripts')!r})\n"
        "import run_clinician_experiment as r\n"
        "r.glob.glob = lambda *a: ['fake.jpg']\n"
        "r.ManifestBuilder.load = lambda *a: pd.DataFrame()\n"
        "r._recording_meta = lambda *a: {}\n"
        "r.load_seg_table = lambda *a: None\n"
        "r._make_callables = lambda *a: (None, None, lambda *a: {}, None)\n"
        "r._input_identity = lambda *a: {}\n"
        "r._free_cuda = lambda: None\n"
        "def fail(*a, **kw):\n    raise RuntimeError('synthetic session failure')\n"
        "r._run_session = fail\n"
        f"raise SystemExit(r.main(['--out', {str(out)!r}, '--conditions', 'A', '--seeds', '0', '--budgets', '1']))\n",
        encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-B", str(script)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "CLINICIAN_EXP_FAILED" in result.stdout
    assert "synthetic session failure" in result.stdout + result.stderr
    assert master.read_text() == "previous valid master\n"
    assert (old_session / "curve.csv").read_text() == "saved session\n"
    logs = list(out.glob("attempts/*/failed.jsonl"))
    assert len(logs) == 1 and json.loads(logs[0].read_text())["session"] == "A_s0"


def test_main_success_reuses_sessions_and_fingerprints_inputs_once(experiment, monkeypatch):
    for stem in experiment.stems:
        (experiment.images / f"{stem}.jpg").write_bytes(b"tiny synthetic image")
    monkeypatch.setattr(runner.ManifestBuilder, "load", lambda *a: pd.DataFrame())
    monkeypatch.setattr(runner, "_recording_meta", lambda *a: {})
    monkeypatch.setattr(runner, "load_seg_table", lambda *a: None)
    monkeypatch.setattr(runner, "_make_callables", lambda *a: experiment.callables)
    monkeypatch.setattr(runner, "_free_cuda", lambda: None)
    original = runner._input_identity
    identity_calls = []
    def identity(*args):
        identity_calls.append(1)
        return original(*args)
    monkeypatch.setattr(runner, "_input_identity", identity)
    argv = ["--out", str(experiment.root), "--conditions", "A", "ref",
            "--seeds", "0", "--budgets", ".2", ".4", ".6", ".8"]
    assert runner.main(argv) == 0
    assert len(identity_calls) == 1 and len(experiment.calls) == 8
    master = experiment.root / "clinician_curve.csv"
    before = master.read_bytes()
    assert runner.main(argv) == 0
    assert len(identity_calls) == 2 and len(experiment.calls) == 8
    assert master.read_bytes() == before
    assert len(list(experiment.root.glob("attempts/*/previous_clinician_curve.csv"))) == 1


def test_first_download_and_repeat_use_same_base_checkpoint_identity(experiment, monkeypatch, tmp_path):
    checkpoint = tmp_path / "yolo11n-pose.pt"
    downloads = []
    def download(alias):
        downloads.append(alias)
        checkpoint.write_bytes(b"synthetic checkpoint fixture")
        return str(checkpoint)
    real_resolve = runner._resolve_checkpoint
    monkeypatch.setattr(runner, "_resolve_checkpoint",
                        lambda weights: real_resolve(weights, download_fn=download))
    monkeypatch.chdir(tmp_path)
    callables = runner._make_callables([], {}, "temporal", 2, 320, 2, 3)
    first_config = callables[2].experiment_config
    experiment.run(run_config=first_config)
    repeated = runner._make_callables([], {}, "temporal", 2, 320, 2, 3)
    assert repeated[2].experiment_config == first_config
    experiment.run(run_config=repeated[2].experiment_config)
    assert len(downloads) == 1 and len(experiment.calls) == 4
    checkpoint.write_bytes(b"different checkpoint fixture")
    changed = runner._make_callables([], {}, "temporal", 2, 320, 2, 3)
    with pytest.raises(runner.SessionReuseError, match="fingerprint differs"):
        experiment.run(run_config=changed[2].experiment_config)
    assert len(experiment.calls) == 4


@pytest.mark.parametrize("operation", ["remove", "empty", "replace"])
def test_cache_requires_intact_completed_weights(experiment, operation):
    experiment.run()
    weights = experiment.session / "round_3" / "mock-best.pt"
    if operation == "remove":
        weights.unlink()
    else:
        weights.write_bytes(b"" if operation == "empty" else b"different weights")
    with pytest.raises(runner.SessionReuseError, match="incomplete/invalid"):
        experiment.run()
    assert len(experiment.calls) == 4
