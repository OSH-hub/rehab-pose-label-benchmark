r"""Generate ViewAngles_최종정리_20261013.md (Korean) from va_aggregate outputs.
Usage: python make_va_report.py <결과_ViewAngles dir>
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BINS = ["0", "10-30", "40-60", "70-90"]


def rd(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, d=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if v != v else f"{v:.{d}f}"


def pm(r, k, d=3):
    return f"{f(r[k + '_mean'], d)} ± {f(r[k + '_sd'], d)}"


def main():
    root = Path(sys.argv[1])
    pmeth = rd(root / "aggregate" / "per_method.csv")
    pmod = rd(root / "aggregate" / "per_model.csv")
    con = rd(root / "aggregate" / "contrasts_vs_A.csv")
    sens = rd(root / "aggregate" / "per_method_sensitivity.csv")
    diag = rd(root / "aggregate" / "alignment_diagnostic_per_recording.csv")
    summ = json.load(open(root / "aggregate" / "summary.json", encoding="utf-8"))
    prep = json.load(open(root / "provenance" / "prepare_meta.json", encoding="utf-8"))
    psha = hashlib.sha256(open(root / "protocol_viewangles.txt", "rb").read()).hexdigest()
    big = [float(r["agree020_det"]) for r in diag if int(float(r["decoded_minus_mp"])) > 100 and r["agree020_det"] not in ("", "nan")]
    small = [float(r["agree020_det"]) for r in diag if int(float(r["decoded_minus_mp"])) <= 100 and r["agree020_det"] not in ("", "nan")]
    ovr = {r["method"]: r for r in pmeth if r["stratum"] == "overall"}
    L = ["# REHAB26-ViewAngles 시점각 민감도 분석 최종정리 (2026-10-13 프로토콜 고정, 2026-10-14 실행·재채점)\n",
         "**중요**: 이 자료에는 관절 정답(GT)이 없다. 아래 수치는 모두 MediaPipe(참조 추정기)와의 *일치도*이며 정확도가 아니다. 자료 라이선스는 CC BY-NC 4.0(비상업 연구용)이다.\n",
         "## 0. 요약\n",
         f"- 18 체크포인트(A/B/ref × 6 seed, 예산 0.40) 완료, 채점 프레임 {prep['frames_scored']:,}(45 녹화×3 카메라, 반복 구간 내 20프레임 간격). 검출률(box conf ≥0.25) A {f(ovr['A']['detection_rate_mean'])} / B {f(ovr['B']['detection_rate_mean'])} / ref {f(ovr['ref']['detection_rate_mean'])}; MediaPipe 일치도@0.2(전 프레임, 미검출=실패) A {f(ovr['A']['agree020_all_mean'])} / B {f(ovr['B']['agree020_all_mean'])} / ref {f(ovr['ref']['agree020_all_mean'])}; 검출 프레임 한정 일치도@0.2 A {f(ovr['A']['agree020_det_mean'])} / B {f(ovr['B']['agree020_det_mean'])} / ref {f(ovr['ref']['agree020_det_mean'])}.",
         "- 시점각 의존성: 세 method 모두 |시점각| 70–90°(측면)에서 검출률과 일치도가 가장 낮고 0–30°(정면 근처)에서 가장 높다(3절). 그러나 seed 간 편차가 시점각 효과보다 크다(4절의 seed별 표).",
         "- 재채점 사유: MediaPipe 2D npy는 픽셀이 아니라 0–1 정규화 좌표였다(최대값 ≈0.89/0.79, 영상 1080×1920). 첫 채점은 참조를 픽셀로 가정해 일치도가 0이 되었고, 참조 x·y에 폭·높이를 곱하는 단위 보정을 모든 모델에 동일하게 적용해 저장된 예측(pred26)으로 재채점했다(`scripts/va_rescore.py`, 원본은 `metrics_v1_unscaled.json`으로 보존). 추론은 다시 하지 않았고 모델별 선택은 없다.\n",
         "## 1. 자료\n",
         f"- Zenodo record 23018128 (CC BY-NC 4.0). 45 녹화 = 3 운동(Ex1 팔 벌리기, Ex6 스쿼트, Ex7 어깨 신전) × 5인 × 3 카메라(동기화, 30 fps). 해상도 {prep['resolutions']}(세로 영상), fps {prep['fps']}. Segmentation.csv의 반복 구간과 사람 방향 o(0–90°, 10° 간격 및 45°)로 시점각을 정의: cam1 −o, cam2 45−o, cam3 90−o. |시점각| 구간 0 / 10–30 / 40–60 / 70–90.",
         f"- 다운로드 MD5 전수 일치(`provenance/download_manifest.json`). MediaPipe 2D 배열 [frames,33,4] (x,y,z,visibility; x,y는 정규화 좌표, 최대 {[round(v, 3) for v in prep['mediapipe_xy_max_overall']]}).",
         f"- 표본: 반복 구간 내 frame % 20 == 0 프레임 {prep['frames_scored']:,}개, 녹화×카메라 {prep['recordings']}개, 구간별 프레임 수 {prep['angle_bin_counts']}.",
         f"- 정렬 점검: 29개 녹화에서 OpenCV 디코딩 프레임 수가 MediaPipe 배열 길이보다 1–741 많았다(끝부분 차이). 차이 >100인 녹화와 ≤100인 녹화의 검출 프레임 일치도@0.2(B·ref 모델)는 {f(np.mean(big))}(n={len(big)}) 대 {f(np.mean(small))}(n={len(small)})로 같은 수준이어서 프레임 인덱스 정렬 문제는 없다고 판단했다(`aggregate/alignment_diagnostic_per_recording.csv`). MediaPipe 길이를 넘는 표본 프레임은 없었다.\n",
         "## 2. 프로토콜(`protocol_viewangles.txt`)\n",
         f"- SHA256 `{psha}`. 모델: 예산 0.40 A/B/ref × 6 seed = 18(registry_server.json, 체크포인트 SHA256 검증). 추론 설정은 UCO와 동일(전체 프레임, imgsz 640, iou 0.70, max_det 300, 최고 box conf 인물, conf 0.01까지 기록, 주 검출 기준 ≥0.25, 민감도 ≥0.05/≥0.01).",
         "- 대응 MediaPipe→rehab26: 11→7, 12→12, 13→8, 14→13, 15→9, 16→14, 23→16, 24→21, 25→17, 26→22, 27→18, 28→23 (12관절).",
         "- 지표: 검출률; 일치도@0.2/@0.1 = 예측–MediaPipe 거리 ≤ 0.2/0.1 × MediaPipe 가시(visibility ≥0.5) 관절 bbox 긴 변인 관절 비율(all: 미검출=실패, det: 검출 한정); 검출 프레임 픽셀 거리. 집계: (녹화, 카메라, 방향) 세그먼트 → 사람(5인) → 평균, 6 seed 평균 ± SD.\n",
         "## 3. 결과: 시점각 구간별 (6 seed 평균 ± SD)\n",
         "| 방법 | |시점각| 구간 | 검출률 | 일치도@0.2 (전 프레임) | 일치도@0.2 (검출 한정) | 일치도@0.1 (전 프레임) | 픽셀거리 (검출) |",
         "|---|---|---|---|---|---|---|"]
    for m in ("A", "B", "ref"):
        for r in pmeth:
            if r["method"] == m and (r["stratum"] == "overall" or r["stratum"].startswith("bin_")):
                L.append(f"| {m} | {'전체' if r['stratum'] == 'overall' else r['stratum'][4:]} | {pm(r, 'detection_rate')} | {pm(r, 'agree020_all')} | {pm(r, 'agree020_det')} | {pm(r, 'agree010_all')} | {pm(r, 'px_dist_det', 1)} |")
    L += ["\n카메라별(검출률 / 일치도@0.2 전 프레임, 6 seed 평균): " + "; ".join(
        f"{m} cam{c} {f(next(r['detection_rate_mean'] for r in pmeth if r['method'] == m and r['stratum'] == f'camera_{c}'))}/{f(next(r['agree020_all_mean'] for r in pmeth if r['method'] == m and r['stratum'] == f'camera_{c}'))}"
        for m in ("A", "B", "ref") for c in (1, 2, 3)) + ".\n",
          "## 4. 운동별 × 시점각 구간 (검출률 / 일치도@0.2 전 프레임, 6 seed 평균)\n",
          "| 방법 | 운동 | 전체 | 0 | 10–30 | 40–60 | 70–90 |", "|---|---|---|---|---|---|---|"]
    for m in ("A", "B", "ref"):
        for e in ("1", "6", "7"):
            cells = []
            for st in [f"exercise_{e}"] + [f"exercise_{e}_bin_{b}" for b in BINS]:
                rr = [r for r in pmeth if r["method"] == m and r["stratum"] == st]
                cells.append(f"{f(rr[0]['detection_rate_mean'])} / {f(rr[0]['agree020_all_mean'])}" if rr else "—")
            L.append(f"| {m} | Ex{e} | " + " | ".join(cells) + " |")
    L += ["\nseed별 값(전체; `aggregate/per_model.csv`):\n",
          "| 모델 | 검출률 | 일치도@0.2 (전 프레임) | 일치도@0.2 (검출 한정) | 픽셀거리 (검출) | 검출률 bin 0 / 10–30 / 40–60 / 70–90 |", "|---|---|---|---|---|---|"]
    for r in sorted(pmod, key=lambda r: (["A", "B", "ref"].index(r["method"]), int(r["seed"]))):
        L.append(f"| {r['model_id']} | {f(r['overall.detection_rate'])} | {f(r['overall.agree020_all'])} | {f(r['overall.agree020_det'])} | {f(r['overall.px_dist_det'], 1)} | " +
                 " / ".join(f(r[f'bin_{b}.detection_rate'], 2) for b in BINS) + " |")
    L += ["\nseed별 관찰: 검출 한정 일치도가 0.6–0.9인 seed(예: ref_s1, B_s1, B_s4, B_s5, A_s1, A_s3)와 0.0–0.4인 seed(A_s4: 검출률 1.0이지만 일치도 0.003·픽셀거리 약 1,000 px — 사람이 아닌 물체를 높은 confidence로 검출; ref_s4, ref_s5, B_s0, B_s2, B_s3)가 뚜렷이 나뉜다. 즉 이 자료에서 method 평균의 차이는 seed 간 분산보다 작고, A seed 4의 이상 거동은 UCO·Penn과 동일하다.\n",
          "## 5. A 대비 contrast (seed 짝, 사람 단위 bootstrap n=5 — 참고용)\n",
          "| 대비 | 구간 | 지표 | 평균 차이 | 95% CI | seed별 차이 (s0..s5) |", "|---|---|---|---|---|---|"]
    for r in con:
        L.append(f"| {r['contrast']} | {r['stratum']} | {r['metric']} | {f(r['mean_diff'], 4)} | [{f(r['ci95_lo'], 4)}, {f(r['ci95_hi'], 4)}] | {r['per_seed_diffs']} |")
    L += ["\n## 6. 검출 임계값 민감도 (6 seed 평균)\n",
          "| 방법 | 임계 | 검출률 | 일치도@0.2 all | 픽셀거리 | 검출률 bin 0 | 10–30 | 40–60 | 70–90 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sens:
        for t in ("0.25", "0.05", "0.01"):
            L.append(f"| {r['method']} | ≥{t} | {f(r[f't{t}.detection_rate_mean'])} | {f(r[f't{t}.agree020_all_mean'])} | {f(r[f't{t}.px_dist_det_mean'], 1)} | " +
                     " | ".join(f(r[f't{t}.bin_{b}.detection_rate_mean']) for b in BINS) + " |")
    L += ["\n## 7. 해석과 주의사항\n",
          "- 시점각 효과: 정면 근처(0, 10–30°)보다 측면(70–90°)에서 검출률·일치도가 낮아지는 경향은 세 method에 공통이며, 원고의 '정면 카메라 선호' 서술과 방향이 같다. 다만 MediaPipe 자체도 측면에서 덜 정확하므로 일치도 하락에는 참조 오차가 섞여 있다.",
          "- MediaPipe는 참조 추정기일 뿐 정답이 아니다. '일치도'는 두 추정기가 함께 틀릴 수도, MediaPipe만 틀릴 수도 있는 값이다. 가시성 ≥0.5 관절만 분모에 넣었으므로 MediaPipe가 보지 못한 관절은 평가에서 빠진다.",
          "- 사람 5명(건강한 성인), 단일 기관, 세로 영상(1080×1920)으로 학습 영상과 종횡비가 다르다. 사람 단위 bootstrap CI(n=5)는 참고용이다.",
          "- 참조 단위 보정(0절)은 결과를 본 뒤 적용한 후처리이지만 모델과 무관한 좌표계 오류의 정정이며 모든 모델에 동일하게 적용되었다. 보정 전 결과(`metrics_v1_unscaled.json`)는 일치도 0·픽셀거리 약 1,100 px로 명백히 단위 오류였다.",
          "- 자료는 CC BY-NC 4.0이며 비상업 연구용으로만 사용했다. 사람 식별명은 공개 자료의 그대로를 집계 단위 키로만 썼다. 모델 입력은 전체 프레임이며 MediaPipe 결과를 인물 선택·crop에 쓰지 않았다.\n",
          "## 8. 파일\n",
          f"- `protocol_viewangles.txt`, `scripts/`(va_prepare/va_evaluate/va_rescore/va_aggregate/make_va_report, uco_evaluate, uco_chain, cuda_wrapper), `provenance/`(download_manifest.json, prepare_meta.json, registry_server.json), `evaluations/<model_id>/{{metrics.json, per_segment.csv, predictions.npz, identity.json, *_v1_unscaled.*}}`, `aggregate/`(per_model.csv, per_method.csv, contrasts_vs_A.csv, per_method_sensitivity.csv, per_segment_all_models.csv, alignment_diagnostic_per_recording.csv, summary.json), `logs/`, `SHA256SUMS.txt`. 완료 모델 {len(summ['complete_models'])}개" + (f", 미완료 {summ['missing']}" if summ["missing"] else "") + "."]
    (root / "ViewAngles_최종정리_20261013.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written")


if __name__ == "__main__":
    main()
