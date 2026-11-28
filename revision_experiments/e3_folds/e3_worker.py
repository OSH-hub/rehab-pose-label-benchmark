import argparse, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

REPO = Path(r"RehabAL-pose")
WORK = Path(r"rehabal_work\e3_folds")
SCRIPTS = Path(r"rehabal_work")
ORIG = REPO / "outputs" / "experiments" / "clinician_i4_uniform30"
CLAIMS = WORK / "claims"
LOGS = WORK / "logs"
CONDS = ["A", "B", "ref"]
SEEDS = [0, 1]


def tag_of(cond, seed):
    return f"{cond}_s{seed}" + ("_sig0" if cond == "B" else "")


def job_list():
    jobs = []
    for seed in SEEDS:
        for cond in CONDS:
            jobs.append({"id": f"fold5_{tag_of(cond, seed)}_eval", "kind": "eval5",
                         "fold": 5, "cond": cond, "seed": seed, "tag": tag_of(cond, seed)})
    for fold in (1, 2, 3, 4):
        for seed in SEEDS:
            for cond in CONDS:
                jobs.append({"id": f"fold{fold}_{tag_of(cond, seed)}", "kind": "train",
                             "fold": fold, "cond": cond, "seed": seed, "tag": tag_of(cond, seed)})
    return jobs


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


LOGFILE = None


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    if LOGFILE is not None:
        with open(LOGFILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def run(cmd, gpu, logfile):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    with open(logfile, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== [{now()}] gpu{gpu} $ {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT)
    return p.returncode


def claim(job_id, gpu, ignore):
    CLAIMS.mkdir(parents=True, exist_ok=True)
    if (CLAIMS / f"{job_id}.done").exists():
        return False
    path = CLAIMS / f"{job_id}.claim"
    if ignore and path.exists() and not (CLAIMS / f"{job_id}.done").exists():
        try:
            owner = json.loads(path.read_text(encoding="utf-8")).get("pid")
        except Exception:  # noqa: BLE001
            owner = None
        import psutil
        if owner is not None and psutil.pid_exists(int(owner)):
            return False
        path.unlink()
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"gpu": gpu, "pid": os.getpid(), "time": now()}, f)
    return True


def mark(job_id, status, info):
    (CLAIMS / f"{job_id}.{status}").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--ignore-claims", action="store_true",
                    help="re-claim jobs that have a stale .claim but no .done (mop-up)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these job ids")
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    global LOGFILE
    LOGFILE = LOGS / f"worker_gpu{args.gpu}.log"
    py = sys.executable
    log(f"worker start gpu={args.gpu} pid={os.getpid()} ignore_claims={args.ignore_claims}")
    n_done = 0
    for job in job_list():
        jid = job["id"]
        if args.only and jid not in args.only:
            continue
        if not claim(jid, args.gpu, args.ignore_claims):
            continue
        t0 = time.time()
        logfile = LOGS / f"{jid}.log"
        info = {"gpu": args.gpu, "start": now(), "attempts": []}
        if job["kind"] == "train":
            sess_dir = WORK / f"fold{job['fold']}" / job["tag"]
            log(f"START {jid} (train) -> {sess_dir}")
            base = [py, "-u", str(SCRIPTS / "e3_session.py"), "--fold", str(job["fold"]),
                    "--condition", job["cond"], "--seed", str(job["seed"])]
            rc = run(base, args.gpu, logfile)
            info["attempts"].append({"rc": rc, "restart": False, "sec": round(time.time() - t0)})
            if rc != 0:
                log(f"  {jid} session rc={rc}; retrying once with --restart-incomplete")
                t1 = time.time()
                rc = run(base + ["--restart-incomplete"], args.gpu, logfile)
                info["attempts"].append({"rc": rc, "restart": True, "sec": round(time.time() - t1)})
            if rc != 0 or not (sess_dir / "completion.json").is_file():
                info["end"] = now()
                info["error"] = f"session failed rc={rc}"
                mark(jid, "failed", info)
                log(f"FAILED {jid} rc={rc}")
                continue
        else:
            sess_dir = ORIG / job["tag"]
            log(f"START {jid} (eval-only of main grid) <- {sess_dir}")
        rows = WORK / f"fold{job['fold']}" / job["tag"] / "e3_eval_rows.csv"
        t2 = time.time()
        rc = run([py, "-u", str(SCRIPTS / "e3_eval.py"), "--fold", str(job["fold"]),
                  "--session-dir", str(sess_dir), "--condition", job["cond"],
                  "--seed", str(job["seed"]), "--out", str(rows)], args.gpu, logfile)
        info["eval_rc"] = rc
        info["eval_sec"] = round(time.time() - t2)
        if rc != 0:
            log(f"  {jid} eval rc={rc}; retrying once")
            rc = run([py, "-u", str(SCRIPTS / "e3_eval.py"), "--fold", str(job["fold"]),
                      "--session-dir", str(sess_dir), "--condition", job["cond"],
                      "--seed", str(job["seed"]), "--out", str(rows)], args.gpu, logfile)
            info["eval_rc_retry"] = rc
        info["end"] = now()
        info["total_sec"] = round(time.time() - t0)
        mark(jid, "done" if rc == 0 else "failed", info)
        log(f"{'DONE' if rc == 0 else 'FAILED(eval)'} {jid} in {(time.time() - t0) / 3600:.2f} h")
        n_done += 1
        rc = run([py, "-u", str(SCRIPTS / "e3_aggregate.py")], args.gpu, LOGS / "aggregate.log")
        if rc != 0:
            log(f"  aggregate rc={rc} (non-fatal)")
    log(f"worker gpu{args.gpu} finished; jobs processed={n_done}")
    print("E3_WORKER_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
