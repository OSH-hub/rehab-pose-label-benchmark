from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,math,argparse
from pypdf import PdfReader
from PIL import Image
OUT=Path(__file__).absolute().parent.parent
parser=argparse.ArgumentParser();parser.add_argument('--replace-existing',action='store_true');args=parser.parse_args()
p=json.loads((OUT/'analysis_payload.json').read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_new(name,text):
    target=OUT/name
    if target.exists():
        previous=target.read_text(encoding='utf-8-sig')
        if not args.replace_existing:raise RuntimeError('Existing output preserved')
    with target.open('w' if target.exists() else 'x',encoding='utf-8' if name.endswith('.json') else 'utf-8-sig',newline='') as f:f.write(text)
def table(rows):
    return '\n'.join(f"{r['label']}: AUC {r['normalized_auc_map50_95_mean']:.4f} ± {r['normalized_auc_map50_95_population_sd']:.4f}; 40% mAP50-95 {r['endpoint_map50_95_mean']:.4f} ± {r['endpoint_map50_95_population_sd']:.4f}" for r in rows)
s3=p['tables']['i3_summary'];s7=p['tables']['i7_summary']
idx3={r['series']:r for r in s3}
delta_auc=idx3['A_conf']['normalized_auc_map50_95_mean']-idx3['A0']['normalized_auc_map50_95_mean']
delta_endpoint=idx3['A_conf']['endpoint_map50_95_mean']-idx3['A0']['endpoint_map50_95_mean']
ko=f'''기존 I3/I7 결과 재사용: 회고 보조분석

범위와 계산
이 폴더는 이미 수행된 REHAB24-6 Camera17의 YOLO11n-pose 실험 CSV를 재정리한 보조분석이다. 새로운 학습·예측·외부평가 결과가 아니다. 원본과 서버에 보존된 결과, 학습 설정, 선택 목록, 완료 기록 및 체크포인트 파일을 별도로 감사한 뒤 수치 계산에는 감사된 CSV만 사용했다. 기존 원고와 새로운 5조건·6seed 주분석은 수정하거나 합치지 않았다. Penn Action 및 신규 비교군 성능 파일을 읽지 않았다.

주 지표는 원 데이터 내부 검증 pose mAP50-95다. 각 seed의 5%, 10%, 20%, 40% accepted-label budget 곡선을 사다리꼴로 적분하고 구간 폭 0.35로 나누었다. 40% 지표는 마지막 예산의 mAP50-95다. 모든 수치는 0~1 척도이며 평균 ± 모집단 표준편차(ddof=0), 기록된 학습 seed 0~5의 6개 값이다. 표준편차는 신뢰구간이 아니다. 보조 CSV에는 mAP50도 별도 열로 보존했으며 그림과 아래 표는 mAP50-95를 사용한다. Penn Action의 PCK와 혼용할 수 없다. 가설검정·유의확률·다중비교 검정은 수행하지 않았다.

표 S-I3. 기존 신뢰도 기준 비교군의 회고 비교 (기존 결과 재사용)
{table(s3)}

A_conf는 5% GT 초기화 이후 예측된 26개 키포인트의 평균 신뢰도가 0.5 이상인 프레임을 받아들이는 GT-free confidence gate다. GT OKS에 의존하는 I1 selective correction oracle과 다르며, 완전한 FixMatch 재현으로 표기해서는 안 된다. B는 좌표 잡음 sigma=0인 GT-derived simulated correction과 segmentation-based rejection을 함께 사용하는 전체 적응 정책이다. 따라서 B 비교는 교정만의 고립된 효과가 아니라 전체 정책 효과를 나타낸다.
A_conf-A0 평균차는 정규화 AUC {delta_auc:.4f}, 40% mAP50-95 {delta_endpoint:.4f}다. 기록된 평균은 A_conf에서 높지만 여섯 실행의 변동을 함께 제시한다. 이 기술통계만으로 유의한 개선, 임상적 효용 또는 독립 데이터 일반화를 주장하지 않는다. B와 ref의 결과도 같은 내부 검증 자료에서 얻어진 기존 결과다.

표 S-I7. full-GT 라벨 조건에서 선택 전략의 회고 민감도 (기존 결과 재사용)
{table(s7)}

I7의 네 선택 전략은 모두 full-GT ref 라벨 조건이다. 따라서 선택 전략 민감도에 관한 보조분석이며 A/B의 교정 정책 효과에 관한 직접 비교로 취급하지 않는다. Temporal (anchor)은 기존 ref 결과를 문맥상 한 번 더 제시한 것이며 새로운 반복 실험이 아니다. i3_summary의 ref와 i7_summary의 temporal_anchor는 동일한 여섯 기록이다.
실제 호출 경로에서는 score()를 내림차순 정렬해 선택한다. 'coreset'이라는 원 폴더명에 해당하는 구현은 pool 임베딩의 중심점에서 먼 정도를 사용하는 centroid-distance proxy다. greedy k-center 선택 구현은 이 호출 경로에서 실행되지 않았다. 논문에서는 표준 Core-set을 재현했다고 쓰지 않는다.
Random 선택기의 난수 생성기에는 학습 seed가 전달되지 않았다. 따라서 여섯 개 seed 열은 기록된 학습 실행을 나타내며 acquisition 난수까지 재현 가능한 여섯 반복이라는 뜻이 아니다. 선택 목록은 남아 있지만 seed만으로 같은 선택을 재생할 수 있다고 주장하지 않는다. 이 때문에 네 선택기의 우열에 관한 강한 결론을 피한다.

재사용 근거와 한계
기록된 train args(416px, batch 16, 30 epochs, nano), 예산별 라벨 수 418/837/1673/3347, 같은 초기 선택 목록, 6seed×4예산 완료 및 CSV/state 일치가 감사에서 확인되었다. 원래 실행 당시의 불변 소스·입력 manifest는 없어 bitwise 재현성을 입증하지 못한다. 초기 체크포인트 파일 바이트 해시는 일부 다르고 텐서 단위 동일성은 확인하지 않았다. n_proposed가 빈 기록을 0으로 채우거나 작업 비용으로 해석하지 않았다.

행 수와 중복 주의
I3 seed별 표 24행, I7 seed별 표 30행, 요약표 각각 4행과 5행이다. 예산별 수치 표 216행에는 temporal_anchor로 재표시한 기존 ref의 24행이 포함된다. 그러므로 이 파일의 216행을 216개의 독립 실험으로 세지 않는다. 회고 보조분석과 새로운 30모델 외부평가/120행 source-domain 주분석은 별도 자료다.

파일
- i3_per_seed.csv / i3_summary.csv: I3 비교의 seed별 수치와 평균·모집단 SD.
- i7_per_seed.csv / i7_summary.csv: I7 네 선택 전략과 기존 temporal anchor.
- audited_source_budget_curves.csv: 두 표에서 사용한 예산별 수치. source_group을 통해 출처를 추적한다.
- historical_i3_confidence_comparator.png/.pdf 및 historical_i7_acquisition_sensitivity.png/.pdf: 각 두 패널의 과학적 보조그림. PNG는 300 dpi, PDF는 벡터 형식이다.
- English_figure_captions.txt: 영문 캡션 및 권장 결과 문장.
- analysis_payload.json: 원 CSV 경로·SHA256·정의·모든 테이블.
- code/: 계산, CSV 작성, 그림 작성, 문서·검증 실행 코드.
- artifact_validation.json: CSV round-trip, PDF 구조, 파일 SHA256 및 시각 검토 기록.

재실행
code/reanalyze_historical.py는 번들 Python으로 계산한다. code/write_tables.mjs는 번들 Node와 @oai/artifact-tool로 CSV를 작성한다. code/plot_historical.py는 이미 설치된 matplotlib를 이용해 CPU로 그림만 생성한다. code/write_notes_and_verify.py는 번들 Python으로 문서와 검증 기록을 생성한다. 기존 파일은 기본적으로 덮어쓰지 않으며 plot_historical.py의 --replace-existing은 검토한 그림을 수리하는 경우에만 명시적으로 사용한다. 원본 CSV/원고/동결된 학습·평가 코드는 수정하지 않는다.
'''
write_new('한글해석_기존결과재사용.txt',ko)
eng='''Supplementary Figure S-I3. Retrospective confidence-gate comparison using historical result reuse. Panel A shows normalized pose mAP50-95 AUC over accepted-label budgets of 5%, 10%, 20%, and 40%, computed by trapezoidal integration divided by 0.35. Panel B shows pose mAP50-95 at the 40% budget. The historical YOLO11n-pose conditions are original self-training (A0), mean-keypoint-confidence-gated self-training (A_conf; threshold 0.5 after shared 5% GT initialization), GT-derived correction plus segmentation-based rejection (B; zero coordinate noise), and full ground truth (ref). Blue dots represent six recorded training runs; black diamonds and bars show the mean and population standard deviation (ddof=0). These reused results are from internal validation on REHAB24-6 Camera17, not from the new Penn Action external evaluation. No hypothesis tests were performed. A_conf is a GT-free frame gate, not the GT-OKS oracle or a full FixMatch reproduction. B represents the total adaptive policy of correction plus rejection, not the isolated effect of correction alone.

Supplementary Figure S-I7. Retrospective acquisition sensitivity under full-GT labeling using historical result reuse. Panels show the same normalized mAP50-95 AUC and 40% endpoint definitions as Figure S-I3. The four historical acquisition runs (random, least-confidence score, centroid-distance proxy, and RehabAL score) all use the full-GT ref labeling condition. The temporal anchor repeats the historical ref result from Figure S-I3 and is not an additional independent experiment. Blue dots represent six recorded training runs; black diamonds and bars show the mean and population standard deviation (ddof=0), not confidence intervals. The originally named coreset condition follows a centroid-distance scoring path and does not execute greedy k-center selection. The random acquisition RNG was not fixed by the recorded training seed; saved selected-frame lists are retained, but the seed alone does not reproduce acquisition. These source-domain internal-validation mAP results are descriptive and are kept separate from the new five-condition evaluation and Penn Action PCK.

Supplementary Table S-I3. Historical result reuse: mean and population standard deviation over six recorded training seeds for the confidence-gate comparison. Normalized AUC is the trapezoidal integral of source-domain validation pose mAP50-95 across budgets 0.05, 0.10, 0.20, and 0.40 divided by 0.35. The endpoint is mAP50-95 at budget 0.40. The historical A0, zero-noise B, and ref anchors are reported alongside A_conf; these are not newly trained results.

Supplementary Table S-I7. Historical result reuse: descriptive full-GT acquisition sensitivity across six recorded training runs. The temporal anchor is the same ref curve used in Table S-I3. The four acquisition strategies retain the implementation qualifications described in Figure S-I7. SD uses ddof=0; no statistical significance is inferred.

Suggested results paragraph (descriptive, historical only):
'''+f'''In a retrospective reanalysis of the audited historical YOLO11n-pose runs, confidence-gated self-training yielded a normalized source-validation mAP50-95 AUC of {idx3['A_conf']['normalized_auc_map50_95_mean']:.4f} +/- {idx3['A_conf']['normalized_auc_map50_95_population_sd']:.4f}, compared with {idx3['A0']['normalized_auc_map50_95_mean']:.4f} +/- {idx3['A0']['normalized_auc_map50_95_population_sd']:.4f} for the original self-training arm (mean +/- population SD over six recorded training runs). The descriptive mean difference was {delta_auc:.4f}. At the 40% budget, the corresponding values were {idx3['A_conf']['endpoint_map50_95_mean']:.4f} +/- {idx3['A_conf']['endpoint_map50_95_population_sd']:.4f} and {idx3['A0']['endpoint_map50_95_mean']:.4f} +/- {idx3['A0']['endpoint_map50_95_population_sd']:.4f}. These reused internal-validation results do not establish statistical significance or external generalization. Historical full-GT acquisition runs are reported separately, with the centroid-distance proxy and unseeded random-acquisition limitations made explicit.

Provenance qualification: Saved configurations, selected-frame lists, completed training records, metrics, and checkpoint-file hashes were audited. Immutable manifests of the exact historical source and input data were unavailable, and equality of initial-checkpoint tensors was not verified. Empty n_proposed fields were not converted to zeros or interpreted as annotation cost.
'''
write_new('English_figure_captions.txt',eng)
checks=[]
for name,expected in p['tables'].items():
    path=OUT/(name+'.csv')
    with path.open(encoding='utf-8-sig',newline='') as f:actual=list(csv.DictReader(f))
    assert len(actual)==len(expected)
    for row,ref in zip(actual,expected):
        assert set(row)==set(ref)
        for key,value in ref.items():
            got=row[key]
            if value is None:assert got==''
            elif isinstance(value,bool):assert got.lower()==str(value).lower()
            elif isinstance(value,(int,float)):assert math.isclose(float(got),value,rel_tol=0,abs_tol=1e-15)
            else:assert got==value
    checks.append({'file':path.name,'rows':len(actual),'roundtrip_verified':True})
media=[]
for stem in ['historical_i3_confidence_comparator','historical_i7_acquisition_sensitivity']:
    with Image.open(OUT/(stem+'.png')) as im:
        assert im.size==(3540,1860)
        dpi=im.info.get('dpi');assert dpi and abs(dpi[0]-300)<.1
        size=im.size
    reader=PdfReader(OUT/(stem+'.pdf'));assert len(reader.pages)==1
    extracted=reader.pages[0].extract_text()
    assert 'Historical result reuse' in extracted and 'population SD' in extracted
    assert 'Camera17' in extracted and 'Penn Action PCK' in extracted
    media.append({'stem':stem,'png_pixels':list(size),'png_dpi':dpi,'pdf_pages':1,'pdf_text_labels_verified':True,
       'visual_review':'PNG viewed at full-page scale; all labels, ticks, points, error bars, legends and footnotes readable after spacing repair. PDF uses same Matplotlib figure canvas.'})
assert all(sha(Path(source['path']))==source['sha256'] for source in p['source_csvs'].values())
files=[]
for path in sorted([*OUT.glob('*'),*OUT.joinpath('code').glob('*')]):
    if path.is_file() and path.name!='artifact_validation.json':files.append({'relative_path':path.relative_to(OUT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'verified_historical_only',
    'csv_checks':checks,'figures':media,'source_csv_hashes_rechecked':True,
    'new_or_penn_results_accessed':False,'historical_result_reuse':True,
    'raw_curve_rows':216,'unique_historical_curve_rows_after_ref_anchor_deduplication':192,
    'ref_temporal_anchor_is_same_six_runs':True,'files':files}
write_new('artifact_validation.json',json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps({'status':report['status'],'csv_files':len(checks),'figure_pairs':len(media),'hashed_files':len(files)},ensure_ascii=False))
