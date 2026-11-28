import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import run_control as c


def completed():
    return [m for m in c.ORDER if (c.HERE / "evaluations" / m / "metrics.json").exists()]


def main():
    if (c.HERE / "DONE.txt").exists():
        print((c.HERE / "DONE.txt").read_text(encoding="utf-8"), flush=True)
        return
    preflight = c.read_json(c.HERE / "preflight.json")
    assert c.sha(c.REGISTRY) == preflight["registry_sha256"]
    assert c.sha(c.CODE / "uco_evaluate.py") == preflight["control_evaluator_sha256"]
    assert c.sha(c.HERE / "protocol_416_control.txt") == preflight["control_protocol_sha256"]
    registry = c.read_json(c.REGISTRY)
    entries = {x["model_id"]: x for x in registry["models"]}
    for check in preflight["weight_checks"]:
        assert c.sha(Path(entries[check["model_id"]]["checkpoint"])) == check["checkpoint_sha256"]
    for folder in (c.HERE / "evaluations").iterdir():
        if not folder.is_dir():
            continue
        identity_path = folder / "identity.json"
        if identity_path.exists():
            identity = c.read_json(identity_path)["identity"]
            assert identity["evaluator_sha256"] == preflight["control_evaluator_sha256"]
            assert identity["protocol_sha256"] == preflight["control_protocol_sha256"]
            assert identity["manifest_sha256"] == preflight["manifest_sha256"]
            assert identity["gt_npz_sha256"] == preflight["gt_npz_sha256"]
            assert identity["inference"]["imgsz"] == 416
            assert identity["model_entry"]["model_id"] == folder.name
    shards = {1: c.ORDER[:18], 0: c.ORDER[18:]}
    assert not set(shards[0]) & set(shards[1]) and set(shards[0] + shards[1]) == set(c.ORDER)
    c.state("evaluating_two_gpus", physical_gpus=[0, 1], subset_sizes={str(k): len(v) for k, v in shards.items()},
            completed_models=len(completed()), parallel_started_at_utc=c.now(), error=None, traceback=None)
    commands = []
    running = []
    logs = []
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for gpu, mids in shards.items():
        args = [sys.executable, "-u", str(c.CODE / "cuda_wrapper.py"), str(c.CODE / "uco_evaluate.py"),
                "--prepared", str(c.OLD / "prepared"), "--registry", str(c.REGISTRY),
                "--protocol", str(c.HERE / "protocol_416_control.txt"), "--out", str(c.HERE / "evaluations"),
                "--device", "0", "--batch", "32", "--chunk-size", "512", "--loader-threads", "8",
                "--models", ",".join(mids)]
        env = os.environ.copy()
        env.update(KMP_DUPLICATE_LIB_OK="TRUE", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(gpu))
        log_path = c.HERE / "logs" / f"evaluation_gpu{gpu}_{stamp}.log"
        log = log_path.open("x", encoding="utf-8")
        proc = subprocess.Popen(args, cwd=c.HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
        running.append((gpu, proc))
        logs.append(log)
        commands.append({"physical_gpu": gpu, "model_ids": mids, "argv": args, "pid": proc.pid, "log": str(log_path)})
    c.json_new(c.HERE / f"parallel_execution_{stamp}.json", {"created_at_utc": c.now(), "workers": commands})
    try:
        last_count = -1
        last_report = 0
        while True:
            done = completed()
            poll = {str(g): p.poll() for g, p in running}
            if len(done) != last_count or time.monotonic() - last_report >= 60:
                c.state("evaluating_two_gpus", completed_models=len(done), completed_model_ids=done, worker_exit_codes=poll)
                last_count, last_report = len(done), time.monotonic()
            if all(v is not None for v in poll.values()):
                break
            time.sleep(5)
        assert all(v == 0 for v in poll.values()), poll
        assert set(done) == set(c.ORDER), done
    finally:
        for log in logs:
            log.close()
    c.state("aggregating", completed_models=36)
    existing = list((c.HERE / "aggregate").glob("*"))
    for p in existing:
        if p.is_file():
            p.read_bytes()
    c.run([sys.executable, str(c.CODE / "uco_aggregate.py"), "--evaluations", str(c.HERE / "evaluations"),
           "--registry", str(c.REGISTRY), "--out", str(c.HERE / "aggregate"), "--require-all"], "aggregate.log")
    c.state("comparing")
    c.compare(preflight)
    c.json_new(c.HERE / "comparison" / "parallel_execution_note.json", {
        "created_at_utc": c.now(), "physical_gpus": [0, 1], "logical_device": 0,
        "subsets": shards, "note": "Disjoint subsets used identical evaluator/settings. Original UCO 640 results were read-only."
    })
    c.state("complete", completed_models=36, completed_at_utc=c.now())
    c.write_new(c.HERE / "DONE.txt", c.now() + "\n36/36 models, aggregation and matched comparison complete.\n")


if __name__ == "__main__":
    lock = c.HERE / "runner.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(json.dumps({"pid": os.getpid(), "started_at_utc": c.now()}))
    try:
        main()
    except BaseException as exc:
        c.state("failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        lock.read_text(encoding="utf-8")
        lock.unlink()
