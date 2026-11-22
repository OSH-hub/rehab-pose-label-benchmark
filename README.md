# Label quality and label quantity in rehabilitation pose estimation: benchmark code

Code and aggregate results for the manuscript

> You-Sun Jeong, Myung-Mi Kim and Seunghong Oh. *A Benchmark Protocol for Label Quality and Label Quantity in Video-Based Rehabilitation Pose Estimation.*

The benchmark compares three ways of supplying training labels to a YOLO pose model under a fixed frame budget:

- **A** – pseudo-label self-training after a shared 5% reference-labelled initialization,
- **B** – simulated reference-derived correction with rejection,
- **ref** – reference labelling.

Condition B is simulated from the dataset reference annotations. It does not measure annotation by clinicians or annotation time.

## What is in this repository

| Path | Content |
|---|---|
| `src/rehabal_pose/` | Core package: REHAB24-6 parser, frame extraction, YOLO-pose training and inference, the budgeted labelling loop, pose and kinematic metrics, statistics and plots |
| `scripts/` | Entry points for the REHAB24-6 experiments (training loop, downstream kinematics, analyses, figures) |
| `configs/` | Dataset, model and experiment settings |
| `tests/` | Unit tests and a synthetic fixture generator (no real data) |
| `revision_experiments/` | Scripts for the additional experiments reported in the manuscript (see below) |
| `results/` | Aggregate result tables behind the reported figures and tables |

`revision_experiments/` is organized by experiment:

| Folder | Experiment |
|---|---|
| `e3_folds/` | Five participant-pair folds on REHAB24-6 |
| `e1_sameframe/` | Fixed-frame label-source substitution |
| `baselines/` | Strengthened self-training arms, summaries and the supplementary tables |
| `crossview_c18/` | Second camera view (Camera18) and downstream kinematic errors |
| `uco/` | External evaluation on UCO Physical Rehabilitation at 640 and 416 pixels |
| `penn_action/` | External evaluation on Penn Action |
| `keraal/`, `viewangles/` | Agreement with reference estimators on KERAAL and REHAB26-ViewAngles |
| `figures/` | Figure and table generation from the result files |
| `retrospective_comparators/` | Re-analysis of earlier comparator runs |

## What is not included

- No dataset files, video frames, reference annotations or participant-level tables. Obtain each dataset from its provider under the provider's terms (`DATA_SOURCES.md`).
- No trained or pretrained weights and no per-frame predictions.
- No machine-specific launch or file-transfer scripts. The experiment scripts were run on a Windows workstation with two NVIDIA RTX 3090 GPUs.

## Installation

```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements-analysis.txt
pip install -e . --no-deps
python tests/fixtures/rehab24_6_mock/make_fixtures.py
python -m pytest -q -m "not slow"
```

Training and inference additionally need `requirements-training.txt` (PyTorch, Ultralytics) with a PyTorch build that matches your CUDA version. `TESTED_ENVIRONMENT.txt` lists the versions used.

## Running the experiments

1. Download REHAB24-6 and place it under `data/rehab24_6_raw/` (`Segmentation.csv` and the Camera17 30-fps videos).
2. Build the manifest with `Rehab24_6Parser` (`configs/datasets/rehab24_6.yaml`) and the YOLO dataset with `build_rehab_yolo.py`.
3. Run the budgeted comparison:

```
python scripts/run_clinician_isolated.py --seeds 0 1 2 3 4 5 --conditions A B ref --sigmas 0 --gpus 0 1 --epochs 30 --epochs-round0 30 --weights yolo11n-pose.pt --out outputs/experiments/clinician_i4_uniform30
python scripts/analyze_clinician_experiment.py --csv outputs/experiments/clinician_i4_uniform30/clinician_curve.csv --metric map50_95 --sigma 0
```

The scripts in `revision_experiments/` take their input and output locations from command-line options or from path constants near the top of each file. The defaults are relative placeholders (`RehabAL-pose`, `rehabal_work`, `rehabal_exp`); set them to your own locations. `figures/make_figures.py` reads `REHABAL_REVISION_ROOT` and `REHABAL_CODE_ROOT`. Some report generators write their text output in Korean.

## Aggregate results

`results/` holds method-level and model-level summaries, paired contrasts with bootstrap intervals, and the data files behind the figures. The `weights` column in `e1_sameframe` and `e3_folds` records the relative location of each checkpoint in the original run directory; the checkpoints themselves are not distributed.

## License

The authors' code is released under the GNU Affero General Public License v3.0 (`LICENSE`). Third-party packages keep their own licenses (`THIRD_PARTY_NOTICES.md`). Datasets are not covered by this license.

## Citation

Please cite the manuscript above. `CITATION.cff` will be updated with the article details after publication.
