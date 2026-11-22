"""Quick sanity-check: print in_rep / occluded rates on a 400-stem sample.

Run from the repo root:
    python scripts/check_seg_rates.py

No GPU required. Reads the real Segmentation.csv and the existing YOLO pool.
"""

from __future__ import annotations

import glob
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from rehabal_pose.review.clinician_loop import load_seg_table, _seg_flags

SEG_CSV = _ROOT / "data" / "rehab24_6_raw" / "Segmentation.csv"
POOL_DIR = _ROOT / "outputs" / "rehab24_6_yolo" / "images" / "train"
SAMPLE_N = 400
SEED = 42


def main() -> None:
    seg = load_seg_table(SEG_CSV)
    if seg is None:
        print(f"ERROR: Segmentation.csv not found at {SEG_CSV}")
        sys.exit(1)
    print(f"Seg rows loaded: {len(seg)}")

    all_imgs = sorted(glob.glob(str(POOL_DIR / "*.jpg")))
    if not all_imgs:
        print(f"ERROR: no images found under {POOL_DIR}")
        sys.exit(1)
    stems = [Path(p).stem for p in all_imgs]
    print(f"Total pool stems: {len(stems)}")

    rng = random.Random(SEED)
    sample = rng.sample(stems, min(SAMPLE_N, len(stems)))
    print(f"Sample size: {len(sample)}")

    flags = [_seg_flags(s, seg) for s in sample]
    in_rep_count = sum(1 for ir, _ in flags if ir)
    occ_count = sum(1 for _, oc in flags if oc)

    print(f"in_rep  : {in_rep_count}/{len(sample)} = {in_rep_count/len(sample)*100:.1f}%")
    print(f"occluded: {occ_count}/{len(sample)} = {occ_count/len(sample)*100:.1f}%")

    # Video_id breakdown for the unknown-video rate
    video_ids_in_seg = set(seg["video_id"].dropna().unique())
    unknown = [s for s in sample if s.split("_rep")[0] not in video_ids_in_seg]
    if unknown:
        print(f"Unknown video_ids in sample: {len(unknown)} "
              f"(will always return in_rep=True, occluded=False)")
        print(f"  e.g. {unknown[:3]}")


if __name__ == "__main__":
    main()
