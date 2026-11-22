r"""E3: build Camera17 YOLO-pose fold datasets by re-splitting the existing
``outputs/rehab24_6_yolo`` (train+val = all 9356 frames, frame_stride 20) by subject.

Why re-split instead of re-extracting: ``build_yolo_pose_dataset`` selects frames
per recording as ``range(0, n_frames, frame_stride)`` and writes one GT label per
frame, independent of the train/val assignment; only the destination folder
depends on ``val_subjects``. So a fold dataset built from the same manifest with
``val_subjects=<pair>`` is byte-identical (images and labels) to re-arranging the
existing frames -- and avoids decoder non-determinism across re-extractions.

Images are hard-linked (same drive), labels copied. Writes data.yaml with the
repo's own writer, verifies counts, and writes val-only eval yamls for folds 1-5.

Usage (repo root):  python rehabal_work\e3_build_folds.py
"""
import os, re, sys, json, shutil
from pathlib import Path

REPO = Path(r"RehabAL-pose")
WORK = Path(r"rehabal_work\e3_folds")
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
from rehabal_pose.data import ManifestBuilder  # noqa: E402
from rehabal_pose.data.yolo_dataset import write_pose_data_yaml, rehab26_flip_idx  # noqa: E402

FOLDS = {1: ["1", "2"], 2: ["3", "4"], 3: ["5", "6"], 4: ["7", "8"], 5: ["9", "10"]}
SRC = REPO / "outputs" / "rehab24_6_yolo"
TOTAL = 9356
STEM = re.compile(r"^(?P<sid>.+)_f(?P<idx>\d+)$")


def dataset_dir(k):
    return SRC if k == 5 else REPO / "outputs" / f"rehab24_6_yolo_fold{k}"


def main():
    df = ManifestBuilder.load(str(REPO / "outputs" / "rehab24_6_real" / "manifest.parquet"))
    sid2subj = {str(r.sample_id): str(r.subject_id) for r in df.itertuples()}

    # inventory of all source frames (train + val) with subject
    items = []  # (stem, subject, img_path, lbl_path)
    for split in ("train", "val"):
        for img in sorted((SRC / "images" / split).glob("*.jpg")):
            m = STEM.match(img.stem)
            subj = sid2subj[m.group("sid")]
            lbl = SRC / "labels" / split / f"{img.stem}.txt"
            if not lbl.is_file():
                raise RuntimeError(f"source label missing: {lbl}")
            items.append((img.stem, subj, img, lbl))
    if len(items) != TOTAL:
        raise RuntimeError(f"source frame count {len(items)} != {TOTAL}")
    stems = [it[0] for it in items]
    if len(set(stems)) != TOTAL:
        raise RuntimeError("duplicate stems across train/val in source")
    print("source frames", len(items), flush=True)

    report = {}
    for k in (1, 2, 3, 4):
        out = dataset_dir(k)
        val_set = set(FOLDS[k])
        ok_marker = out / "_E3_BUILD_OK.json"
        if ok_marker.is_file():
            prev = json.loads(ok_marker.read_text(encoding="utf-8"))
            ntr = len(list((out / "images" / "train").glob("*.jpg")))
            nva = len(list((out / "images" / "val").glob("*.jpg")))
            if prev.get("train") == ntr and prev.get("val") == nva and ntr + nva == TOTAL:
                print(f"fold{k}: already built (train={ntr} val={nva}); skip", flush=True)
                report[k] = prev
                continue
            print(f"fold{k}: marker present but counts differ; rebuilding", flush=True)
            shutil.rmtree(out)
        for split in ("train", "val"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        counts = {"train": 0, "val": 0}
        per_subj = {"train": {}, "val": {}}
        for stem, subj, img, lbl in items:
            split = "val" if subj in val_set else "train"
            dst_img = out / "images" / split / f"{stem}.jpg"
            dst_lbl = out / "labels" / split / f"{stem}.txt"
            if not dst_img.exists():
                try:
                    os.link(img, dst_img)
                except OSError:
                    shutil.copyfile(img, dst_img)
            if not dst_lbl.exists():
                shutil.copyfile(lbl, dst_lbl)
            counts[split] += 1
            per_subj[split][subj] = per_subj[split].get(subj, 0) + 1
        write_pose_data_yaml(out, out / "images" / "train", out / "images" / "val")
        # verify
        ntr = len(list((out / "images" / "train").glob("*.jpg")))
        nva = len(list((out / "images" / "val").glob("*.jpg")))
        ltr = len(list((out / "labels" / "train").glob("*.txt")))
        lva = len(list((out / "labels" / "val").glob("*.txt")))
        assert ntr + nva == TOTAL, (ntr, nva)
        assert ltr == ntr and lva == nva, (ltr, ntr, lva, nva)
        assert set(per_subj["val"]) == val_set, per_subj["val"]
        for p in (out / "images" / "val").glob("*.jpg"):
            if not (out / "labels" / "val" / f"{p.stem}.txt").is_file():
                raise RuntimeError(f"val label missing for {p.stem}")
        info = {"fold": k, "val_subjects": FOLDS[k], "train": ntr, "val": nva,
                "per_subject": per_subj, "source": str(SRC), "images": "hardlink"}
        ok_marker.write_text(json.dumps(info, indent=2), encoding="utf-8")
        report[k] = info
        print(f"fold{k}: val_subjects={FOLDS[k]} train={ntr} val={nva} sum={ntr + nva} "
              f"val_per_subject={per_subj['val']}", flush=True)

    # val-only eval yamls (absolute path) for all five folds
    ydir = WORK / "yamls"
    ydir.mkdir(parents=True, exist_ok=True)
    flip = rehab26_flip_idx()
    for k in (1, 2, 3, 4, 5):
        root = dataset_dir(k).resolve().as_posix()
        lines = [f"path: {root}", "train: images/val", "val: images/val", "",
                 "kpt_shape: [26, 3]", f"flip_idx: {flip}", "", "names:", "  0: person", ""]
        (ydir / f"fold{k}_val.yaml").write_text("\n".join(lines), encoding="utf-8")
    (WORK / "folds_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("E3_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
