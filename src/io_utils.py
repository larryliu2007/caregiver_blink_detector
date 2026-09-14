"""Trial-workbook validation, cache checks, and output summaries."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


REQUIRED_TRIAL_COLUMNS = [
    "video_file",
    "participant_id",
    "trial_id",
    "onset_s",
    "offset_s",
]


def _clean_identifier(value):
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError("participant_id and trial_id cannot be blank")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value.strip() if isinstance(value, str) else value


def load_trials(workbook_path):
    """Load and validate the five-column trial workbook."""
    workbook_path = Path(workbook_path)
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Trial workbook not found: {workbook_path}")

    trials = pd.read_excel(workbook_path)
    trials.columns = [str(column).strip() for column in trials.columns]
    missing = [column for column in REQUIRED_TRIAL_COLUMNS if column not in trials]
    if missing:
        raise ValueError(f"trials.xlsx is missing required columns: {missing}")

    trials = trials[REQUIRED_TRIAL_COLUMNS].copy()
    if trials.empty:
        raise ValueError("trials.xlsx contains no trial rows")

    trials["video_file"] = trials.video_file.astype("string").str.strip()
    blank_video = trials.video_file.isna() | trials.video_file.eq("")
    if blank_video.any():
        raise ValueError("video_file cannot be blank")
    invalid_path = trials.video_file.map(lambda value: Path(str(value)).name != str(value))
    if invalid_path.any():
        rows = (np.flatnonzero(invalid_path.to_numpy()) + 2).tolist()
        raise ValueError(f"video_file must be a file name, not a path (Excel rows {rows})")

    trials["participant_id"] = trials.participant_id.map(_clean_identifier)
    trials["trial_id"] = trials.trial_id.map(_clean_identifier)
    trials["onset_s"] = pd.to_numeric(trials.onset_s, errors="coerce")
    trials["offset_s"] = pd.to_numeric(trials.offset_s, errors="coerce")

    one_blank = trials.onset_s.isna() ^ trials.offset_s.isna()
    if one_blank.any():
        rows = (np.flatnonzero(one_blank.to_numpy()) + 2).tolist()
        raise ValueError(
            "onset_s and offset_s must either both be filled or both be blank "
            f"(Excel rows {rows})"
        )

    specified = trials.onset_s.notna()
    invalid_time = specified & (
        (trials.onset_s < 0) | (trials.offset_s <= trials.onset_s)
    )
    if invalid_time.any():
        rows = (np.flatnonzero(invalid_time.to_numpy()) + 2).tolist()
        raise ValueError(
            "Each specified trial needs onset_s >= 0 and offset_s > onset_s "
            f"(Excel rows {rows})"
        )

    duplicated = trials.duplicated(["participant_id", "trial_id"], keep=False)
    if duplicated.any():
        rows = (np.flatnonzero(duplicated.to_numpy()) + 2).tolist()
        raise ValueError(
            f"participant_id + trial_id must be unique (Excel rows {rows})"
        )

    trials.insert(0, "pipeline_trial_id", np.arange(1, len(trials) + 1))
    return trials


def video_duration_seconds(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if (
        not np.isfinite(fps)
        or fps <= 0
        or not np.isfinite(frame_count)
        or frame_count <= 0
    ):
        raise ValueError(f"Could not determine video duration: {video_path}")
    return frame_count / fps


def resolve_video_paths(trials, videos_dir):
    """Locate videos and expand paired blank times to the whole video."""
    videos_dir = Path(videos_dir)
    resolved = trials.copy()
    video_paths = {}
    durations = {}

    for video_file in resolved.video_file.unique():
        video_path = videos_dir / str(video_file)
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video '{video_file}' was not found in {videos_dir}"
            )
        video_paths[str(video_file)] = video_path
        durations[str(video_file)] = video_duration_seconds(video_path)

    whole_video = resolved.onset_s.isna() & resolved.offset_s.isna()
    resolved.loc[whole_video, "onset_s"] = 0.0
    resolved.loc[whole_video, "offset_s"] = resolved.loc[
        whole_video, "video_file"
    ].map(durations)

    beyond_end = resolved.apply(
        lambda row: float(row.offset_s) > durations[str(row.video_file)] + 0.05,
        axis=1,
    )
    if beyond_end.any():
        rows = (np.flatnonzero(beyond_end.to_numpy()) + 2).tolist()
        raise ValueError(f"Trial offset exceeds the video duration (Excel rows {rows})")

    resolved["onset_s"] = resolved.onset_s.astype(float)
    resolved["offset_s"] = resolved.offset_s.astype(float)
    return resolved, video_paths


def cache_matches_trials(features, trials):
    """Return True only when a cached video has the same requested trials."""
    if features.empty:
        return False
    required = {
        "pipeline_trial_id",
        "participant_id",
        "trial_id",
        "trial_onset_ms",
        "trial_offset_ms",
    }
    if not required.issubset(features.columns):
        return False

    cached = features[list(required)].drop_duplicates("pipeline_trial_id")
    expected = trials[
        ["pipeline_trial_id", "participant_id", "trial_id", "onset_s", "offset_s"]
    ].copy()
    expected["trial_onset_ms"] = expected.onset_s * 1000
    expected["trial_offset_ms"] = expected.offset_s * 1000
    comparison = expected.merge(
        cached, on="pipeline_trial_id", suffixes=("_new", "_cache")
    )
    if len(comparison) != len(expected):
        return False

    same_ids = (
        comparison.participant_id_new.astype(str).eq(
            comparison.participant_id_cache.astype(str)
        )
        & comparison.trial_id_new.astype(str).eq(
            comparison.trial_id_cache.astype(str)
        )
    ).all()
    same_times = np.allclose(
        comparison[["trial_onset_ms_new", "trial_offset_ms_new"]],
        comparison[["trial_onset_ms_cache", "trial_offset_ms_cache"]].to_numpy(),
        rtol=0,
        atol=0.01,
    )
    return bool(same_ids and same_times)


def processing_summary(trials, features, events):
    """Build one easy-to-read processing row per requested trial."""
    rows = []
    for trial in trials.itertuples(index=False):
        trial_features = features[
            features.pipeline_trial_id == trial.pipeline_trial_id
        ]
        trial_events = events[events.pipeline_trial_id == trial.pipeline_trial_id]
        valid_percent = (
            100.0 * trial_features.tracking_valid.astype(bool).mean()
            if len(trial_features)
            else 0.0
        )
        rows.append(
            {
                "video_file": trial.video_file,
                "participant_id": trial.participant_id,
                "trial_id": trial.trial_id,
                "trial_minutes": (trial.offset_s - trial.onset_s) / 60,
                "detected_blinks": len(trial_events),
                "tracking_valid_percent": valid_percent,
            }
        )
    return pd.DataFrame(rows)
