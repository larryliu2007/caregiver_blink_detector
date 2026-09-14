# Final caregiver blink-detection pipeline

This folder is the small deployment version of the validated caregiver workflow. It reads requested video intervals from Excel, extracts eye-aspect ratio (EAR), finds sensitive local EAR dips, and uses the frozen caregiver Random Forest to retain likely blinks.

## Files required to run

Keep this entire small folder structure:

```text
final_caregiver_pipeline/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── config.py
├── src/
│   ├── extract_ear.py
│   ├── detect_blinks.py
│   └── io_utils.py
├── models/
│   ├── face_landmarker.task
│   └── caregiver_blink_rf.joblib
├── data/
│   ├── trials.xlsx
│   └── videos/
└── outputs/
```

`README.md` and `.gitignore` are not imported by Python, but should remain: this file contains the instructions and `.gitignore` prevents videos and generated outputs from being committed accidentally. The two files in `models/` are runtime dependencies and must not be removed.

## Validated method

The caregiver method is:

**MediaPipe Face Landmarker → bilateral pixel-coordinate EAR → local EAR-dip candidates → 22-feature Random Forest → threshold 0.35**

The historical PELT/change-point implementation is archived exploratory work and is not the validated final caregiver detector. The caregiver pipeline does use a Random Forest. It does not use infant Q7, selective abstention, dlib, the experimental eye-image CNN, or any infant detector branch.

Timing preserves the historical convention: frame number divided by the video's reported FPS. Trial endpoints are converted to frames and processed inclusively.

## Installation

Python 3.11 is recommended. The preserved environment used for exact historical verification contained the versions in `requirements.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Input

1. Put each MP4 file in `data/videos/`.
2. Edit `data/trials.xlsx` with one row per trial.

Required columns:

| Column | Meaning |
|---|---|
| `video_file` | Exact file name in `data/videos/` |
| `participant_id` | Numeric or text study ID |
| `trial_id` | Numeric or text trial ID |
| `onset_s` | Trial start in seconds |
| `offset_s` | Trial end in seconds |

`onset_s` and `offset_s` may both be blank to process the whole video. Filling only one, using a non-positive interval, exceeding the video duration, or defining overlapping intervals for one video produces a clear error.

Example row:

| video_file | participant_id | trial_id | onset_s | offset_s |
|---|---:|---:|---:|---:|
| caregiver_example.mp4 | example_01 | 1 | 15.30 | 42.10 |

## Run

From this folder:

```bash
python run_pipeline.py
```

The first run extracts EAR and caches it under `outputs/features/`. Later runs reuse a cache only when its trial IDs and times still match `trials.xlsx`.

## Outputs

- `outputs/blink_results.csv` — video, participant, trial, blink onset, offset, midpoint, and RF score.
- `outputs/processing_summary.csv` — trial duration, detected-blink count, and percentage of frames with valid tracking.

## Historical performance

The saved July 8 leave-one-caregiver-out evaluation covered 11 included caregivers, 54 trials, and 2,692 manual blink events. At the frozen RF threshold of 0.35:

- True positives: 2,617
- False negatives: 75
- False positives: 140
- Precision: 0.9492
- Recall: 0.9721
- F1: 0.9605

The threshold maximized aggregate F1 in the saved nine-point sweep; threshold selection was not nested inside each held-out fold. The serialized deployment RF was subsequently fitted on all 3,835 labeled candidates from the same 11-caregiver development subset. It has no separate independent held-out estimate.

## Limitations

- Performance is supported for the historical caregiver development data and evaluation contract, not every camera, pose, population, or recording condition.
- Reliable face landmarks are required. Long tracking gaps remain missing instead of being silently bridged.
- Timing accuracy depends on the FPS reported by OpenCV because that is the validated historical convention.
- Four historical caregiver records were excluded by the validation code; the repository does not document the substantive reason, so no reason is inferred here.
- This pipeline detects blinks; it does not provide a quality gate or manual-review recommendation.

## Privacy

Never commit participant videos, annotations, extracted features, or study-identifying filenames. `.gitignore` excludes `data/videos/*` and `outputs/*` while keeping their `.gitkeep` placeholders.

