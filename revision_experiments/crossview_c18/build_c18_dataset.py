"""Build the Camera18 (transposed, 1080x1920) YOLO-pose dataset of REHAB24-6.

Mirrors build_rehab_yolo.py (c17) exactly: same val subjects (9, 10), same
frame_stride (20), same per-recording frame indices, so c17/c18 validation
frames are paired one-to-one. Writes:
  outputs/rehab24_6_real_c18/manifest.parquet  (+ keypoints/)
  outputs/rehab24_6_yolo_c18/{images,labels}/{train,val}, data.yaml
"""
import glob, os, sys, time
import numpy as np
sys.path.insert(0, "src")
from rehabal_pose.data import rehab24_6 as r246
from rehabal_pose.data import ManifestBuilder, build_yolo_pose_dataset

ROOT = "data/rehab24_6_raw"
OUT_REAL = "outputs/rehab24_6_real_c18"
OUT_YOLO = "outputs/rehab24_6_yolo_c18"

Parser = next(getattr(r246, n) for n in dir(r246) if n.lower().startswith("rehab24_6parser"))
cfg = {
    "dataset": "rehab24_6", "camera": "c18", "camera_id": "c18", "fps": "30",
    "pose2d_dirname": "2d_joints", "pose3d_dirname": "3d_joints",
    "segmentation_name": "Segmentation.csv", "output_dir": OUT_REAL,
    "skeleton": "rehab26", "subject_id_regex": None, "view": "vertical",
}
t0 = time.time()
mp = os.path.join(OUT_REAL, "manifest.parquet")
if not os.path.exists(mp):
    builder = Parser(cfg).parse(ROOT)
    builder.save(mp)
    print("manifest saved", mp, "rows", len(builder), f"{time.time()-t0:.0f}s")
df = ManifestBuilder.load(mp)
rec = df.drop_duplicates("video_id").copy().reset_index(drop=True)
print("recordings:", len(rec), "| subjects:", sorted(rec["subject_id"].dropna().unique()))

def vp(vid):
    hits = glob.glob(f"{ROOT}/videos/*/{vid}-Camera18-30fps-transposed.mp4")
    return hits[0] if hits else None
rec["video_path"] = rec["video_id"].map(lambda v: vp(str(v)))
print("video_path resolved:", rec["video_path"].notna().sum(), "| missing:", rec["video_path"].isna().sum())
rec = rec[rec["video_path"].notna()].reset_index(drop=True)

kpmap = {str(r.sample_id): r.keypoints2d_path for r in rec.itertuples()}
# sanity: c18 coordinate ranges must fit the transposed 1080x1920 frame
a = np.load(kpmap[str(rec.sample_id[0])])
print("kp sample", a.shape, "x max", float(np.nanmax(a[..., 0])), "y max", float(np.nanmax(a[..., 1])))

out = build_yolo_pose_dataset(
    rec, lambda sid: np.load(kpmap[sid]), OUT_YOLO,
    val_subjects=["9", "10"], frame_stride=20,
    image_size_lookup=lambda s: (1080.0, 1920.0),
)
ntr = len(glob.glob(f"{OUT_YOLO}/images/train/*.jpg")); nva = len(glob.glob(f"{OUT_YOLO}/images/val/*.jpg"))
print(f"DATASET_DONE {out} images: train={ntr} val={nva} elapsed {time.time()-t0:.0f}s")
