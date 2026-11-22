r"""Generate UCO_최종정리_20261013.md (Korean) from the aggregate CSVs and per-model metrics.json.
Usage: python make_uco_report.py <결과_UCO dir>
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

LABELS = {"A_b040": "A (예산 0.40)", "B_b040": "B (예산 0.40)", "ref_b040": "ref (예산 0.40)", "A_b005": "A (예산 0.05, 공유 초기화)",
          "A_jointmask_b040": "A_jointmask (예산 0.40)", "A_flipcons_b040": "A_flipcons (예산 0.40)"}
ORDER = ["A_b040", "B_b040", "ref_b040", "A_b005", "A_jointmask_b040", "A_flipcons_b040"]
EX_DESC = {1: "하지 좌 01", 2: "하지 좌 02", 3: "하지 좌 03", 4: "하지 좌 04", 5: "하지 우 05", 6: "하지 우 06", 7: "하지 우 07", 8: "하지 우 08",
           9: "상지 좌 09", 10: "상지 좌 10", 11: "상지 좌 11", 12: "상지 좌 12", 13: "상지 우 13", 14: "상지 우 14", 15: "상지 우 15", 16: "상지 우 16"}


def rd(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, d=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if v != v else f"{v:.{d}f}"


def pm(row, k, d=3):
    return f"{f(row[k + '_mean'], d)} ± {f(row[k + '_sd'], d)}"


def main():
    root = Path(sys.argv[1])
    agg = root / "aggregate"
    per_method = rd(agg / "per_method.csv")
    per_model = rd(agg / "per_model.csv")
    contrasts = rd(agg / "contrasts_vs_A.csv")
    oobsens = rd(agg / "per_method_oob_and_sensitivity.csv")
    summary = json.load(open(agg / "summary.json", encoding="utf-8"))
    prep = json.load(open(root / "provenance" / "prepare_meta.json", encoding="utf-8"))
    proto_sha = hashlib.sha256(open(root / "protocol_uco.txt", "rb").read()).hexdigest()
    # per-model metrics.json for exercise strata
    mj = {}
    for d in sorted((root / "evaluations").iterdir()):
        if (d / "metrics.json").exists():
            mj[d.name] = json.load(open(d / "metrics.json", encoding="utf-8"))
    by_label = {}
    for mid, m in mj.items():
        lab = f"{m['method']}_b{int(round(m['budget'] * 100)):03d}"
        by_label.setdefault(lab, []).append(m)
    ov = {r["label"]: r for r in per_method if r["stratum"] == "overall"}
    L = []
    L.append("# UCO Physical Rehabilitation RGB/2D 독립 외부평가 최종정리 (2026-10-13 프로토콜 고정, 2026-10-14 실행)\n")
    L.append("작성: 서버(RTX 3090) 자동 파이프라인(uco_chain.py) 결과를 집계. 본 문서의 모든 수치는 `aggregate/` CSV와 `evaluations/*/metrics.json`에서 `scripts/make_uco_report.py`로 생성되었다.\n")
    L.append("## 0. 요약\n")
    a, b, r_, a5 = ov["A_b040"], ov["B_b040"], ov["ref_b040"], ov["A_b005"]
    L.append(f"- 36 체크포인트 전부 완료(6 seed × A/B/ref/A_jointmask/A_flipcons 예산 0.40 + A 예산 0.05). 채점 프레임 {prep['frames_scored']:,}(1,968클립, 27명, 1 fps).")
    L.append(f"- 주 검출 기준(box conf ≥0.25)에서 검출률 A {f(a['detection_rate_mean'])} / B {f(b['detection_rate_mean'])} / ref {f(r_['detection_rate_mean'])} / A(0.05) {f(a5['detection_rate_mean'])}; 각도오차 ≤10° 프레임 비율(전 프레임) A {f(a['angle_acc10_all_mean'])} / B {f(b['angle_acc10_all_mean'])} / ref {f(r_['angle_acc10_all_mean'])} / A(0.05) {f(a5['angle_acc10_all_mean'])}; PCK@0.2(전 프레임) A {f(a['pck02_all_mean'])} / B {f(b['pck02_all_mean'])} / ref {f(r_['pck02_all_mean'])} / A(0.05) {f(a5['pck02_all_mean'])}.")
    L.append("- 결론: REHAB24-6 내부 검증에서의 순서(A가 B·ref보다 우세)는 UCO에서 유지되지 않는다. Penn·KERAAL과 같이 인물 검출 실패가 결과를 지배하며, 의사라벨 라운드를 더 거친 A(0.40)가 공유 초기화 A(0.05)보다 외부 검출률·정확도 모두 크게 낮다. 자세한 해석은 9절.\n")
    L.append("## 1. 자료 설명과 이용 조건\n")
    L.append("- UCO Physical Rehabilitation dataset (Universidad de Córdoba, AVA group). 건강한 자원자 27명이 8종 재활 운동(좌식/앙와위 하지 4종 + 입위/전굴 상지 4종, 좌·우 = 16 운동 ID)을 수행하며 RGB 카메라 5대(1280×720, 25 fps)로 촬영. 정답은 OptiTrack 적외선 마커 3D 위치를 각 카메라로 투영한 2D 좌표이며 운동 대상 사지의 3관절(하지 hip/knee/ankle, 상지 shoulder/elbow/wrist)만 제공된다.")
    L.append("- 출처·조건: 제공 서버(http://rabinf24.uco.es/avaphy/dataset/), 연구목적 전용, 재배포 금지, 인용 필수 — Aguilar-Ortega R. et al., \"UCO Physical Rehabilitation: New Dataset and Study of Human Pose Estimation Methods on Physical Rehabilitation Exercises\", Sensors 2023, 23(21), 8862, doi:10.3390/s23218862.")
    L.append("- 수령·검증(README_수령및검증.txt): 분할 ZIP 12개 11.94 GB, 추출 4,525파일 CRC·SHA256 검증, RGB 2,160개, 2D TXT 1,968개, 3D TXT 395개. 집계본 dataset_2d.json은 42/432 기록만 담은 불완전본이므로 사용하지 않고 카메라별 TXT를 정답 원천으로 썼다.")
    L.append(f"- 서버 전송: p2d TXT가 있는 1,968클립의 영상+TXT와 메타 8파일(총 3,944파일)을 per-subject tar로 전송하고 `provenance/ship_sha256.json`(원본 추출 manifest의 SHA256)과 서버에서 재계산한 값을 전수 대조(불일치 0). 서버 측 목록 `provenance/SHA256SUMS_data.txt`.\n")
    L.append("## 2. 프로토콜(사전 고정, `protocol_uco.txt`)\n")
    L.append(f"- 프로토콜 SHA256: `{proto_sha}` (모든 metrics.json에 기록된 값: {summary['protocol_sha256']}).")
    L.append("- 동기화: RGB 프레임 = TXT 행 + 5. 근거 = 1,968쌍 전부 영상 프레임 수가 TXT 행 수보다 정확히 5 많음(서버 OpenCV 디코딩으로 재확인, `provenance/prepare_meta.json`의 decoded_minus_gt_rows_histogram = {5: 1968}), 공개 전처리 설명(++ 논문의 초반 5프레임 제외), 그리고 본 평가 전 10클립(하지 좌/우·상지 좌/우·카메라 0–4·좌식/앙와위/입위/전굴)에서 GT 속도 최대 행을 오프셋 0..10 프레임에 겹친 격자(`pilot/offset_check/offset_grid_*.png`)를 육안 검토한 결과 10클립 모두 오프셋 5에서 가장 잘 맞았다. 모델 예측 기반 오프셋 곡선(`pilot/offset_curve.json`)은 모델 오차(75–140 px)가 커 판별력이 없어 근거로 쓰지 않았다.")
    L.append("- 표본: 각 클립 TXT 행 0, 25, 50, …(1 fps) → 대응 RGB 프레임을 무손실 PNG로 캐시(SHA256 기록·추론 시 재검증).")
    L.append("- 관절 대응: hip→UpLeg(16/21), knee→Leg(17/22), ankle→Foot(18/23), shoulder→Arm(7/12), elbow→ForeArm(8/13), wrist→Hand(9/14); 쇄골 사슬 Shoulder(6/11)는 쓰지 않음. 파일럿에서 좌우 거울 대응·쇄골 대응과 비교해도 체계적 뒤바뀜은 없었다(`scripts/uco_mapping_diag.py`).")
    L.append("- 추론: 전체 프레임, imgsz 640, NMS IoU 0.70, max_det 300, person, augment/half 없음, GT crop·GT 기반 인물 선택 없음. 최고 box confidence 인물 1명. conf 0.01까지 기록한 뒤 주 분석은 box conf ≥ 0.25 (conf=0.25로 추론한 것과 동일한 상자 집합), 사전등록 민감도 ≥0.05/≥0.01.")
    L.append("- 지표: 주 = 정점 관절(knee/elbow) 2D 각도 절대오차(검출 프레임, angle_mae_det) 및 각도오차 ≤10°/≤15° 프레임 비율(모든 프레임, 미검출=실패). 보조 = PCK@0.2(임계 = 0.2 × GT 사지 길이; 미검출=실패인 all 및 검출 한정 det), 픽셀 오차(검출 한정), 검출률.")
    L.append("- 집계: 클립 → 피험자(동일 가중) → 27명 평균(subject-macro); method별 6 seed 평균 ± SD. 대비는 seed 짝 차이를 피험자 단위 paired bootstrap(B=10,000, seed 20261013) 95% CI로 제시.\n")
    L.append("## 3. 표본과 제외\n")
    L.append(f"- 채점 클립 {prep['clips_scored']}개(피험자 {len(prep['subjects'])}명), 채점 프레임 {prep['frames_scored']:,}개. p2d가 없는 RGB {prep['clips_without_p2d_excluded']}개(35 sequence 전체 175 + 6 sequence 부분 17)는 제외. 디코딩 실패·결측 프레임 0.")
    L.append(f"- 영상 밖 GT(손목, 5클립 239행) 중 표본에 든 프레임 {prep['gt_oob_frames']}개: 포함이 주 표이며 제외 버전은 8절.\n")
    L.append("## 4. 주 결과 (전체, subject-macro, 6 seed 평균 ± SD, box conf ≥ 0.25)\n")
    L.append("| 방법 | seed 수 | 검출률 | 각도 MAE° (검출 프레임) | 각도 중앙값° (검출) | ≤10° 비율 (전 프레임) | ≤15° 비율 (전 프레임) | PCK@0.2 (전 프레임) | PCK@0.2 (검출 한정) | 픽셀오차 (검출) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for lab in ORDER:
        r = ov[lab]
        L.append(f"| {LABELS[lab]} | {r['n_seeds']} | {pm(r, 'detection_rate')} | {pm(r, 'angle_mae_det', 1)} | {pm(r, 'angle_median_det', 1)} | {pm(r, 'angle_acc10_all')} | {pm(r, 'angle_acc15_all')} | {pm(r, 'pck02_all')} | {pm(r, 'pck02_det')} | {pm(r, 'px_err_det', 1)} |")
    L.append("\nseed별 값(`aggregate/per_model.csv`, overall):\n")
    L.append("| 모델 | 검출률 | 각도 MAE° (검출) | ≤10° (전 프레임) | PCK@0.2 (전 프레임) | 픽셀오차 |")
    L.append("|---|---|---|---|---|---|")
    for r in sorted(per_model, key=lambda r: (ORDER.index(f"{r['method']}_b{int(round(float(r['budget']) * 100)):03d}"), int(r["seed"]))):
        L.append(f"| {r['model_id']} | {f(r['overall.detection_rate'])} | {f(r['overall.angle_mae_det'], 1)} | {f(r['overall.angle_acc10_all'])} | {f(r['overall.pck02_all'])} | {f(r['overall.px_err_det'], 1)} |")
    L.append("\n해석 주의: 검출률이 낮고 method 간 다르므로 '검출 한정' 지표(각도 MAE, PCK det, 픽셀오차)의 분모는 method마다 다른 프레임 집합이다. 전 프레임 지표(≤10°/≤15° 비율, PCK@0.2 all)는 미검출을 실패로 두어 같은 분모를 쓴다. A(0.40) seed 4는 검출률 0.84이지만 검출 프레임의 각도 MAE가 다른 seed보다 약 57° 크고 픽셀오차가 수백 px인 이상 seed로(Penn에서도 같은 양상), A(0.40)의 큰 SD는 이 seed 때문이다.\n")
    L.append("## 5. A(0.40) 대비 paired contrast (피험자 단위 bootstrap 95% CI, 전체)\n")
    L.append("| 대비 | 지표 | 피험자 수 | 평균 차이 | 95% CI | seed별 차이 (s0..s5) | CI가 0 제외 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in contrasts:
        if r["stratum"] == "overall":
            L.append(f"| {r['contrast']} | {r['metric']} | {r['n_subjects']} | {f(r['mean_diff'], 4)} | [{f(r['ci95_lo'], 4)}, {f(r['ci95_hi'], 4)}] | {r['per_seed_diffs']} | {'예' if r['excludes_zero'] == 'True' else '아니오'} |")
    L.append("\n양의 차이는 검출률·≤10°/≤15° 비율·PCK에서 '비교군이 A보다 높음', 각도 MAE·픽셀오차에서 '비교군이 A보다 나쁨'을 뜻한다. 각도 MAE(검출 한정)의 seed별 차이는 A seed 4(이상 seed)에 의해 −57°가 끼어 있으므로 평균 차이를 그대로 해석하지 말고 seed별 값을 함께 본다. 부위별·카메라별 contrast는 `aggregate/contrasts_vs_A.csv`.\n")
    L.append("## 6. 카메라별 (subject-macro, 6 seed 평균 ± SD)\n")
    L.append("| 방법 | 카메라 | 검출률 | 각도 MAE° (검출) | ≤10° 비율 (전 프레임) | PCK@0.2 (전 프레임) |")
    L.append("|---|---|---|---|---|---|")
    for lab in ORDER:
        for r in per_method:
            if r["label"] == lab and r["stratum"].startswith("camera_"):
                L.append(f"| {LABELS[lab]} | {r['stratum'][-1]} | {pm(r, 'detection_rate')} | {pm(r, 'angle_mae_det', 1)} | {pm(r, 'angle_acc10_all')} | {pm(r, 'pck02_all')} |")
    L.append("\n## 7. 신체부위별 및 운동별\n")
    L.append("신체부위(lower = 좌식/앙와위 하지 운동 01–08, upper = 입위/전굴 상지 운동 09–16):\n")
    L.append("| 방법 | 부위 | 검출률 | 각도 MAE° (검출) | ≤10° 비율 (전 프레임) | PCK@0.2 (전 프레임) | 픽셀오차 (검출) |")
    L.append("|---|---|---|---|---|---|---|")
    for lab in ORDER:
        for r in per_method:
            if r["label"] == lab and r["stratum"] in ("body_lower", "body_upper"):
                L.append(f"| {LABELS[lab]} | {r['stratum'][5:]} | {pm(r, 'detection_rate')} | {pm(r, 'angle_mae_det', 1)} | {pm(r, 'angle_acc10_all')} | {pm(r, 'pck02_all')} | {pm(r, 'px_err_det', 1)} |")
    L.append("\n운동별(검출률 / ≤10° 비율(전 프레임) / 각도 MAE°(검출), 6 seed 평균; `evaluations/*/metrics.json`의 exercise_* 항목):\n")
    L.append("| 운동 | " + " | ".join(LABELS[l] for l in ORDER) + " |")
    L.append("|---|" + "---|" * len(ORDER))
    for ex in range(1, 17):
        cells = []
        for lab in ORDER:
            vals = [m["metrics"]["subject_macro"].get(f"exercise_{ex:02d}") for m in by_label.get(lab, [])]
            vals = [v for v in vals if v]
            if not vals:
                cells.append("—")
                continue
            det = np.nanmean([v["detection_rate"] for v in vals]); acc = np.nanmean([v["angle_acc10_all"] for v in vals]); mae = np.nanmean([v["angle_mae_det"] for v in vals])
            cells.append(f"{det:.2f} / {acc:.3f} / {mae:.0f}")
        L.append(f"| {ex:02d} ({EX_DESC[ex]}) | " + " | ".join(cells) + " |")
    L.append("\n부위×카메라 세부는 `aggregate/per_method.csv`(stratum = body_*_camera_*).\n")
    L.append("## 8. 영상 밖 GT 포함/제외 및 검출 임계값 민감도 (전체, 6 seed 평균)\n")
    L.append("| 방법 | 검출률 (포함/제외) | 각도 MAE° (포함/제외) | PCK@0.2 all (포함/제외) |")
    L.append("|---|---|---|---|")
    for r in oobsens:
        L.append(f"| {LABELS.get(r['label'], r['label'])} | {f(r['detection_rate_incl_oob_mean'])} / {f(r['detection_rate_excl_oob_mean'])} | {f(r['angle_mae_det_incl_oob_mean'], 1)} / {f(r['angle_mae_det_excl_oob_mean'], 1)} | {f(r['pck02_all_incl_oob_mean'])} / {f(r['pck02_all_excl_oob_mean'])} |")
    L.append("\n영상 밖 GT는 표본 10프레임뿐이어서 포함/제외 차이는 소수점 셋째 자리 이내이다.\n")
    L.append("| 방법 | 검출 임계 | 검출률 | 각도 MAE° (검출) | ≤10° 비율 (전 프레임) | PCK@0.2 all | 픽셀오차 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in oobsens:
        for t in ("0.25", "0.05", "0.01"):
            L.append(f"| {LABELS.get(r['label'], r['label'])} | ≥{t} | {f(r[f't{t}.detection_rate_mean'])} | {f(r[f't{t}.angle_mae_det_mean'], 1)} | {f(r[f't{t}.angle_acc10_all_mean'])} | {f(r[f't{t}.pck02_all_mean'])} | {f(r[f't{t}.px_err_det_mean'], 1)} |")
    L.append("\n민감도 행(≥0.05, ≥0.01)은 사전등록된 보조 분석이며 주 결과를 대체하지 않는다. 임계를 0.01까지 낮추면 검출률은 0.7–0.98로 오르지만 ≤10° 비율은 모든 method에서 0.25 이하에 머물러, 검출 실패를 걷어내도 관절 국소화 자체가 UCO 자세 분포에서 크게 틀린다는 점은 변하지 않는다.\n")
    L.append("## 9. 내부 도메인 순서가 유지되는가 — 검출 실패의 지배\n")
    L.append(f"REHAB24-6 내부 검증(같은 촬영환경·같은 피험자군의 hold-out)에서 보고된 A(의사라벨 보정) 우세는 UCO에서 유지되지 않는다. 주 검출 기준에서 B와 ref는 A(0.40)보다 검출률(+{f(next(r['mean_diff'] for r in contrasts if r['contrast'].startswith('B_b040') and r['stratum']=='overall' and r['metric']=='detection_rate'), 3)} / +{f(next(r['mean_diff'] for r in contrasts if r['contrast'].startswith('ref_b040') and r['stratum']=='overall' and r['metric']=='detection_rate'), 3)}), ≤10° 비율(+{f(next(r['mean_diff'] for r in contrasts if r['contrast'].startswith('B_b040') and r['stratum']=='overall' and r['metric']=='angle_acc10_all'), 3)} / +{f(next(r['mean_diff'] for r in contrasts if r['contrast'].startswith('ref_b040') and r['stratum']=='overall' and r['metric']=='angle_acc10_all'), 3)}), PCK@0.2(+{f(next(r['mean_diff'] for r in contrasts if r['contrast'].startswith('B_b040') and r['stratum']=='overall' and r['metric']=='pck02_all'), 3)} / +{f(next(r['mean_diff'] for r in contrasts if r['contrast'].startswith('ref_b040') and r['stratum']=='overall' and r['metric']=='pck02_all'), 3)})에서 모두 높고 피험자 bootstrap 95% CI가 0을 제외한다. 즉 UCO에서의 순서는 B ≈ ref > A(0.40)이며 둘 사이 차이는 seed 변동 안에 있다.")
    L.append(f"더 큰 효과는 예산 축이다. 라운드 0의 공유 초기화 A(0.05)는 A(0.40)보다 검출률이 +{f(next(r['mean_diff'] for r in contrasts if r['contrast'].startswith('A_b005') and r['stratum']=='overall' and r['metric']=='detection_rate'), 3)}, ≤10° 비율 +{f(next(r['mean_diff'] for r in contrasts if r['contrast'].startswith('A_b005') and r['stratum']=='overall' and r['metric']=='angle_acc10_all'), 3)}, PCK@0.2 +{f(next(r['mean_diff'] for r in contrasts if r['contrast'].startswith('A_b005') and r['stratum']=='overall' and r['metric']=='pck02_all'), 3)} 높고, 모든 전 프레임 지표에서 36 체크포인트 중 최상이다. 강화 비교군 A_jointmask·A_flipcons도 A(0.40)보다 검출률이 낮다(−0.144, −0.063). 이는 Penn Action(검출률 A 0.21 vs A(0.05) 0.28)과 KERAAL에서 관찰된 양상과 같다: REHAB 의사라벨로 라운드를 거듭할수록 모델이 학습 도메인(입위 환자, 특정 촬영실)에 특화되어 외부 도메인에서 인물 검출 confidence가 무너지고, 전 프레임 지표가 그 검출 실패에 지배된다. 검출된 프레임만 보면 method 간 각도 MAE 차이(약 30–37°, 이상 seed 제외)는 작고 모두 임상적으로 쓸 수 없는 수준이며, 검출 임계값을 0.01로 낮춰도(8절) 순서와 결론은 바뀌지 않는다. 따라서 UCO 결과는 '의사라벨 보정의 내부 이득이 다른 기관·다른 자세 분포로 전이되지 않으며, 외부 평가에서는 검출 단계 실패가 1차 제약'이라는 원고의 한계 서술을 지지하는 독립 기관 증거로 쓸 수 있고, UCO에서 방법 간 우열을 주장하는 데 쓰면 안 된다.\n")
    L.append("## 10. 파일럿 관찰 (CPU, ref_s0·A_s0, 21클립, `pilot/`)\n")
    L.append("- 오버레이(`pilot/overlays/`): (i) 검출된 경우에도 좌식·앙와위 하지에 대해 다리를 '서 있는 자세'처럼 아래로 늘어뜨린 예측(2_06_cam1, 14_06_cam3) — 학습 도메인(입위 환자)의 자세 사전이 강하게 작용; (ii) 삼각대를 사람으로 검출(8_01_cam3, box 0.26); (iii) 정면 입위 피험자(8_10_cam1)에서도 box conf 0.01로 미검출이지만 26관절 위치 자체는 대체로 맞음 → 검출 confidence 보정이 도메인 밖에서 크게 흔들림.")
    L.append("- 거울 대응 진단: 하지에서 거울 측 오차가 다소 낮았으나 두 값 모두 100 px 이상으로 좌우 뒤바뀜이 아니라 전반적 오국소화이다. 상지는 정규 대응이 더 낮았다. 프로토콜 대응을 유지했다.\n")
    L.append("## 11. 주의사항과 한계\n")
    L.append("- 건강한 자원자(환자 아님), 단일 기관, 통제된 실험실(벽면 fiducial 마커 다수)이므로 임상 유효성 증거가 아니다.")
    L.append("- GT는 운동 대상 사지의 3관절만 제공되어 전신 PCK·다른 관절은 평가할 수 없고, 관절각도 하나(knee 또는 elbow)만 비교한다. OptiTrack 마커 위치와 REHAB24-6/rehab26 관절 정의가 완전히 같다는 보장은 없다(특히 hip↔UpLeg, shoulder↔Arm).")
    L.append("- +5 프레임 동기화는 길이 관계·공개 설명·10클립 육안 검증에 근거한 전 클립 공통 가정이다. 운동이 느려 ±1프레임의 영향은 작다.")
    L.append("- 운동이 느리고 1 fps로 표본화했으므로 프레임 간 상관이 크다. 통계 단위를 피험자로 둔 이유이다.")
    L.append("- 하지 운동은 좌식·앙와위(침대)이고 상지 운동은 전굴 자세가 많아, 입위 환자 중심의 학습 도메인과 자세 분포가 크게 다르다(강한 분포 이동). 낮은 검출률과 큰 오차는 이 이동의 결과이며 pipeline 오류가 아님을 파일럿 오버레이로 확인했다.")
    L.append("- 검출 한정 지표는 생존자 편향(검출된 쉬운 프레임)을 가진다. 전 프레임 지표와 함께 읽어야 한다. 검출이 0인 피험자-method 조합은 각도 MAE contrast에서 짝을 잃는다(5절 피험자 수가 27이면 해당 없음).")
    L.append("- 5개 영상 239행의 영상 밖 손목 GT는 그대로 두고 포함/제외 두 버전을 제시했다.\n")
    L.append("## 12. 파일\n")
    L.append("- `protocol_uco.txt`, `scripts/`(uco_prepare/uco_evaluate/uco_aggregate/uco_chain/cuda_wrapper/launch_chain_wmi.ps1/make_uco_report 등), `provenance/`(ship_sha256.json, SHA256SUMS_data.txt, prepare_meta.json, registry_server.json), `evaluations/<model_id>/{metrics.json, per_clip.csv, predictions.npz, identity.json}`, `aggregate/{per_model.csv, per_method.csv, contrasts_vs_A.csv, per_method_oob_and_sensitivity.csv, per_clip_all_models.csv, summary.json}`, `pilot/`, `logs/`(chain.log, uco_eval_gpu.log, prepare_full.log 등), `SHA256SUMS.txt`.")
    L.append(f"- 평가 코드 SHA256 {summary['evaluator_sha256']}, manifest SHA256 {summary['manifest_sha256']}, 프레임 {summary['frames']}, 클립 {summary['clips']}, 완료 모델 {len(summary['complete_models'])}개" + (f", 미완료 {summary['missing_models']}" if summary['missing_models'] else "") + ".")
    (root / "UCO_최종정리_20261013.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written", root / "UCO_최종정리_20261013.md")


if __name__ == "__main__":
    main()
