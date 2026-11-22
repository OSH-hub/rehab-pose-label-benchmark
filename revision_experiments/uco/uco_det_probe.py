r"""Diagnostic: run one checkpoint on a few cached PNG frames with a very low confidence
threshold and print the box confidences / counts, to understand zero-detection clips.
Usage: python uco_det_probe.py <model.pt> <png1> <png2> ...
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(sys.argv[1])
print("names", model.names, "kpt_shape", model.model.model[-1].kpt_shape)
for p in sys.argv[2:]:
    img = cv2.imread(p)
    for conf in (0.25, 0.05, 0.01):
        res = model.predict(source=[img], imgsz=640, conf=conf, iou=0.7, max_det=300, classes=[0], verbose=False, device="cpu")[0]
        bc = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
        print(os.path.basename(os.path.dirname(p)), os.path.basename(p), "conf", conf, "n", len(bc), "top", np.round(bc[:3], 3))
    res = model.predict(source=[img], imgsz=640, conf=0.01, iou=0.7, max_det=300, verbose=False, device="cpu")[0]
    bc = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
    cl = res.boxes.cls.cpu().numpy() if res.boxes is not None else []
    print("   no class filter: n", len(bc), "cls", cl[:5], "conf", np.round(bc[:5], 3))
    if res.boxes is not None and len(bc):
        print("   box xyxy", np.round(res.boxes.xyxy.cpu().numpy()[0], 1))
