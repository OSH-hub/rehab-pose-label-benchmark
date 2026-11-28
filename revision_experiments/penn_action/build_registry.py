import hashlib
import json
import re
import sys
from pathlib import Path

ORIG_ROOT = Path(r"RehabAL-pose\outputs\experiments\clinician_i4_uniform30")
NEW_ROOT = Path(r"rehabal_work\strengthened_suite")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(4 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def original_path(model_id):
    m = re.fullmatch(r"(A|B|ref)_n_s(\d)_b(040|005)", model_id)
    if not m:
        raise ValueError("Unexpected original model_id: " + model_id)
    method, seed, budget = m.group(1), m.group(2), m.group(3)
    rnd = "round_3" if budget == "040" else "round_0"
    folder = {"A": f"A_s{seed}", "B": f"B_s{seed}_sig0", "ref": f"ref_s{seed}"}[method]
    return ORIG_ROOT / folder / rnd / "run" / "train" / "weights" / "best.pt"


def main():
    orig_path, out_path, report_path = (Path(a) for a in sys.argv[1:4])
    with open(orig_path, "r", encoding="utf-8-sig") as f:
        orig = json.load(f)
    models = []
    report = {"original": [], "new": []}
    for entry in orig["models"]:
        ckpt = original_path(entry["model_id"])
        if not ckpt.exists():
            raise FileNotFoundError(str(ckpt))
        digest = sha256_file(ckpt)
        ok = digest == entry["checkpoint_sha256"]
        report["original"].append({"model_id": entry["model_id"], "server_checkpoint": str(ckpt),
                                   "registry_sha256": entry["checkpoint_sha256"], "server_sha256": digest, "match": ok})
        if not ok:
            raise RuntimeError("SHA256 mismatch for " + entry["model_id"])
        new_entry = dict(entry)
        new_entry["checkpoint"] = str(ckpt)
        models.append(new_entry)
    B005_REF = {0: "74bd7a254bd128d77cf885c596b7dc79dd41a87452cc7d77abb49e52c11564b7",
                1: "fc20420dd855dc22a6dedb30f3b5603e191349bbe206ac9250d30b3e055b13e0",
                2: "0c6e3e242f74d11dd748531f144ffa91f47e0f34c21f8788905b45572e06584e",
                3: "a2c82bf8f3c9a03e62c3b0f4e70bf593c5ad02898fe4c9f5a63ef967ce2f724b",
                4: "e92135df34d5a166234d41feccec42f2dd95c9ac80f7c88aa5ff22b7587d806c",
                5: "75f9408d6675834e6888d913f345d4923f4d8ffd747c0e1beb26c20f381542d8"}
    for seed in range(6):
        model_id = f"A_n_s{seed}_b005"
        ckpt = original_path(model_id)
        if not ckpt.exists():
            raise FileNotFoundError(str(ckpt))
        digest = sha256_file(ckpt)
        ok = digest == B005_REF[seed]
        report["original"].append({"model_id": model_id, "server_checkpoint": str(ckpt),
                                   "registry_sha256": B005_REF[seed], "server_sha256": digest, "match": ok,
                                   "note": "reference hash from local models/original_nano file; not in original registry"})
        if not ok:
            raise RuntimeError("SHA256 mismatch for " + model_id)
        models.append({"model_id": model_id, "checkpoint": str(ckpt), "checkpoint_sha256": digest,
                       "method": "A", "seed": seed, "budget": 0.05, "backbone": "yolo11n",
                       "skeleton": "rehab26", "expected_kpt_shape": [26, 3], "provenance": "original_reused"})
    for method in ("A_jointmask", "A_flipcons"):
        for seed in range(6):
            ckpt = NEW_ROOT / f"{method}_s{seed}" / "round_3" / "run" / "train" / "weights" / "best.pt"
            if not ckpt.exists():
                raise FileNotFoundError(str(ckpt))
            digest = sha256_file(ckpt)
            model_id = f"{method}_n_s{seed}_b040"
            models.append({"model_id": model_id, "checkpoint": str(ckpt), "checkpoint_sha256": digest,
                           "method": method, "seed": seed, "budget": 0.4, "backbone": "yolo11n",
                           "skeleton": "rehab26", "expected_kpt_shape": [26, 3], "provenance": "new_training"})
            report["new"].append({"model_id": model_id, "server_checkpoint": str(ckpt), "server_sha256": digest,
                                  "size_bytes": ckpt.stat().st_size})
    ids = [m["model_id"] for m in models]
    if len(ids) != len(set(ids)) or len(ids) != 36:
        raise RuntimeError("Expected 36 unique model ids, got %d" % len(ids))
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"models": models}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"models": len(models), "original_verified": sum(r["match"] for r in report["original"]),
                      "new": len(report["new"])}))


if __name__ == "__main__":
    main()
