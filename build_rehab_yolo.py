import glob, numpy as np
from rehabal_pose.data import ManifestBuilder, build_yolo_pose_dataset

df = ManifestBuilder.load("outputs/rehab24_6_real/manifest.parquet")
rec = df.drop_duplicates("video_id").copy().reset_index(drop=True)
print("recordings:", len(rec), "| subjects:", sorted(rec['subject_id'].dropna().unique()))

def vp(vid):
    hits = glob.glob(f"data/rehab24_6_raw/videos/*/{vid}-Camera17-30fps.mp4")
    return hits[0] if hits else None
rec["video_path"] = rec["video_id"].map(lambda v: vp(str(v)))
missing = rec["video_path"].isna().sum()
print("video_path resolved:", rec["video_path"].notna().sum(), "| missing:", missing)
rec = rec[rec["video_path"].notna()].reset_index(drop=True)

kpmap = {str(r.sample_id): r.keypoints2d_path for r in rec.itertuples()}
def kp_loader(sid): return np.load(kpmap[sid])

out = build_yolo_pose_dataset(
    rec, kp_loader, "outputs/rehab24_6_yolo",
    val_subjects=["9","10"],
    frame_stride=20,
    image_size_lookup=lambda s: (1920.0, 1080.0),
)
print("DATASET_DONE", out)
import os
ntr=len(glob.glob("outputs/rehab24_6_yolo/images/train/*.jpg"))
nva=len(glob.glob("outputs/rehab24_6_yolo/images/val/*.jpg"))
print(f"images: train={ntr} val={nva}")
