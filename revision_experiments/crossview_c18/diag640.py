import os, glob
from ultralytics import YOLO
w = r"RehabAL-pose\outputs\experiments\clinician_i4_uniform30\ref_s0\round_3\run\train\weights\best.pt"
y18 = r"rehabal_work\crossview\c18.yaml"
for imgsz, rect in ((416, True), (416, False), (640, True), (640, False), (512, False), (800, False)):
    r = YOLO(w).val(data=y18, imgsz=imgsz, rect=rect, batch=16, device="0", plots=False, verbose=False, workers=0,
                    project=r"rehabal_work\crossview\runs", name=f"diag_{imgsz}_{rect}", exist_ok=True)
    print(f"c18 ref_s0 r3 imgsz={imgsz} rect={rect}: pose mAP50-95={float(r.pose.map):.4f} mAP50={float(r.pose.map50):.4f} box mAP50={float(r.box.map50):.4f}", flush=True)
imgs = sorted(glob.glob(r"RehabAL-pose\outputs\rehab24_6_yolo_c18\images\val\*.jpg"))[:200]
for imgsz in (416, 640):
    res = YOLO(w).predict(imgs, imgsz=imgsz, conf=0.25, device="0", verbose=False)
    n = sum(1 for x in res if x.boxes is not None and len(x.boxes) > 0)
    print(f"predict c18 first200 imgsz={imgsz}: frames with detection = {n}/200", flush=True)
