"""Readable paths and frozen caregiver detector settings."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
TRIALS_FILE = DATA_DIR / "trials.xlsx"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FEATURES_DIR = OUTPUTS_DIR / "features"

FACE_MODEL_FILE = MODELS_DIR / "face_landmarker.task"
RF_MODEL_FILE = MODELS_DIR / "caregiver_blink_rf.joblib"
BLINK_RESULTS_FILE = OUTPUTS_DIR / "blink_results.csv"
PROCESSING_SUMMARY_FILE = OUTPUTS_DIR / "processing_summary.csv"

REUSE_FEATURE_CACHE = True

# Frozen local EAR-dip candidate settings.
SMOOTH_WINDOW_FRAMES = 3
BASELINE_WINDOW_FRAMES = 30
MIN_DISTANCE_FRAMES = 5
MIN_DROP_RATIO = 0.08
MIN_ABSOLUTE_DROP = 0.0
MIN_PROMINENCE = 0.005
RECOVERY_RATIO = 0.5
MAX_CANDIDATE_WIDTH_FRAMES = 45
MERGE_DISTANCE_MS = 100
MAX_INTERPOLATION_GAP = 3
OCCLUSION_BUFFER_FRAMES = 5
BLINK_SCORE_WINDOW_FRAMES = 2

# Highest-F1 point in the saved July 8 leave-one-caregiver-out sweep.
RF_THRESHOLD = 0.35

