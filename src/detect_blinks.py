"""Frozen caregiver EAR-dip candidates and Random Forest filtering."""

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import config


FEATURE_COLUMNS = [
    "min_ear",
    "baseline_ear",
    "drop",
    "drop_ratio",
    "prominence",
    "width_frames",
    "duration_ms",
    "area_under_drop",
    "closing_slope",
    "opening_slope",
    "blink_score_at_min",
    "blink_score_peak",
    "left_ear_at_min",
    "right_ear_at_min",
    "left_right_asymmetry",
    "near_occlusion",
    "left_right_asymmetry_mean",
    "left_right_asymmetry_max",
    "ear_jitter_window",
    "multi_dip_count",
    "pre_post_ear_shift",
    "blink_score_drop_agreement",
]

CANDIDATE_COLUMNS = [
    "pipeline_trial_id",
    "onset_frame",
    "onset_ms",
    "min_frame",
    "min_ms",
    "end_frame",
    "end_ms",
] + FEATURE_COLUMNS


def _smooth_series(series):
    return (
        series.interpolate(
            limit=config.MAX_INTERPOLATION_GAP, limit_direction="both"
        )
        .rolling(window=config.SMOOTH_WINDOW_FRAMES, center=True, min_periods=1)
        .mean()
    )


def _build_occlusion_mask(trial):
    missing = ~trial.tracking_valid.astype(bool)
    run_id = (missing != missing.shift(fill_value=False)).cumsum()
    run_lengths = missing.groupby(run_id).transform("sum")
    is_occlusion = missing & (run_lengths > config.MAX_INTERPOLATION_GAP)
    near_occlusion = (
        is_occlusion.rolling(
            window=2 * config.OCCLUSION_BUFFER_FRAMES + 1,
            center=True,
            min_periods=1,
        )
        .max()
        .astype(bool)
    )
    return is_occlusion, near_occlusion


def _add_signal_columns(trial):
    trial = trial.copy().reset_index(drop=True)
    is_occlusion, near_occlusion = _build_occlusion_mask(trial)
    trial["near_occlusion"] = near_occlusion.astype(int)

    ear = trial.ear.copy()
    ear[is_occlusion] = np.nan
    trial["ear_clean"] = ear.interpolate(
        limit=config.MAX_INTERPOLATION_GAP, limit_direction="both"
    )
    trial["ear_smooth"] = trial.ear_clean.rolling(
        window=config.SMOOTH_WINDOW_FRAMES, center=True, min_periods=1
    ).mean()
    trial["ear_baseline"] = trial.ear_smooth.rolling(
        window=config.BASELINE_WINDOW_FRAMES, center=True, min_periods=1
    ).median()
    trial["drop"] = trial.ear_baseline - trial.ear_smooth
    trial["drop_ratio"] = trial["drop"] / trial.ear_baseline
    trial["left_ear_smooth"] = _smooth_series(trial.left_ear)
    trial["right_ear_smooth"] = _smooth_series(trial.right_ear)
    trial["blink_score_smooth"] = _smooth_series(trial.blink_score)
    return trial


def _candidate_bounds(trial, minimum_index, baseline, minimum_ear):
    recovery = minimum_ear + config.RECOVERY_RATIO * (baseline - minimum_ear)
    start = minimum_index
    while start > 0 and trial.loc[start, "ear_smooth"] < recovery:
        start -= 1
    end = minimum_index
    while end < len(trial) - 1 and trial.loc[end, "ear_smooth"] < recovery:
        end += 1
    return start, end


def _blink_score_features(trial, index, drop_ratio):
    start = max(0, index - config.BLINK_SCORE_WINDOW_FRAMES)
    end = min(len(trial) - 1, index + config.BLINK_SCORE_WINDOW_FRAMES)
    score_at_minimum = float(trial.loc[index, "blink_score_smooth"])
    peak_score = float(trial.loc[start:end, "blink_score_smooth"].max())
    return {
        "blink_score_at_min": score_at_minimum,
        "blink_score_peak": peak_score,
        "blink_score_drop_agreement": peak_score / max(drop_ratio, 0.001),
    }


def _left_right_features(trial, event, index, baseline):
    left_at_minimum = float(trial.loc[index, "left_ear_smooth"])
    right_at_minimum = float(trial.loc[index, "right_ear_smooth"])
    asymmetry = (
        (event.left_ear_smooth - event.right_ear_smooth).abs()
        / max(baseline, 0.001)
    )
    return {
        "left_ear_at_min": left_at_minimum,
        "right_ear_at_min": right_at_minimum,
        "left_right_asymmetry": abs(left_at_minimum - right_at_minimum)
        / max(baseline, 0.001),
        "left_right_asymmetry_mean": float(asymmetry.mean()),
        "left_right_asymmetry_max": float(asymmetry.max()),
    }


def _context_features(trial, index, start, end, baseline):
    context_start = max(0, index - 15)
    context_end = min(len(trial) - 1, index + 15)
    context_ear = trial.loc[context_start:context_end, "ear_smooth"].copy()
    context_peaks, _ = find_peaks(
        -context_ear.to_numpy(dtype=float),
        distance=3,
        prominence=config.MIN_PROMINENCE,
    )
    pre_mean = trial.loc[
        max(0, start - 10) : max(0, start - 1), "ear_smooth"
    ].mean()
    post_mean = trial.loc[
        min(len(trial) - 1, end + 1) : min(len(trial) - 1, end + 10),
        "ear_smooth",
    ].mean()
    return {
        "ear_jitter_window": float(context_ear.diff().abs().std()),
        "multi_dip_count": int(len(context_peaks)),
        "pre_post_ear_shift": abs(pre_mean - post_mean) / max(baseline, 0.001),
    }


def _candidate_features(trial, pipeline_trial_id, index, start, end, prominence):
    baseline = float(trial.loc[index, "ear_baseline"])
    minimum_ear = float(trial.loc[index, "ear_smooth"])
    onset_ms = float(trial.loc[start, "timestamp_ms"])
    minimum_ms = float(trial.loc[index, "timestamp_ms"])
    end_ms = float(trial.loc[end, "timestamp_ms"])
    event = trial.loc[start:end].copy()

    features = {
        "pipeline_trial_id": pipeline_trial_id,
        "onset_frame": int(trial.loc[start, "frame_index"]),
        "onset_ms": onset_ms,
        "min_frame": int(trial.loc[index, "frame_index"]),
        "min_ms": minimum_ms,
        "end_frame": int(trial.loc[end, "frame_index"]),
        "end_ms": end_ms,
        "min_ear": minimum_ear,
        "baseline_ear": baseline,
        "drop": float(trial.loc[index, "drop"]),
        "drop_ratio": float(trial.loc[index, "drop_ratio"]),
        "prominence": prominence,
        "width_frames": end - start,
        "duration_ms": end_ms - onset_ms,
        "area_under_drop": float(
            np.trapezoid(event["drop"].fillna(0), event["timestamp_ms"])
        ),
        "closing_slope": (baseline - minimum_ear) / max(minimum_ms - onset_ms, 1),
        "opening_slope": (baseline - minimum_ear) / max(end_ms - minimum_ms, 1),
        "near_occlusion": int(trial.loc[index, "near_occlusion"]),
    }
    features.update(_blink_score_features(trial, index, features["drop_ratio"]))
    features.update(_left_right_features(trial, event, index, baseline))
    features.update(_context_features(trial, index, start, end, baseline))
    return features


def _detect_trial_candidates(trial, pipeline_trial_id):
    trial = _add_signal_columns(trial)
    minima, properties = find_peaks(
        -trial.ear_smooth.to_numpy(dtype=float),
        distance=config.MIN_DISTANCE_FRAMES,
        prominence=config.MIN_PROMINENCE,
    )
    rows = []
    for number, index in enumerate(minima):
        baseline = float(trial.loc[index, "ear_baseline"])
        minimum_ear = float(trial.loc[index, "ear_smooth"])
        drop = float(trial.loc[index, "drop"])
        drop_ratio = float(trial.loc[index, "drop_ratio"])
        prominence = float(properties["prominences"][number])

        if pd.isna(drop_ratio) or drop_ratio < config.MIN_DROP_RATIO:
            continue
        if pd.isna(drop) or drop < config.MIN_ABSOLUTE_DROP:
            continue
        if pd.isna(baseline) or pd.isna(minimum_ear) or baseline <= minimum_ear:
            continue

        start, end = _candidate_bounds(trial, index, baseline, minimum_ear)
        width = end - start
        duration = float(
            trial.loc[end, "timestamp_ms"] - trial.loc[start, "timestamp_ms"]
        )
        if width <= 0 or width > config.MAX_CANDIDATE_WIDTH_FRAMES or duration <= 0:
            continue
        rows.append(
            _candidate_features(
                trial, pipeline_trial_id, index, start, end, prominence
            )
        )
    return rows


def _merge_close_candidates(candidates):
    if candidates.empty:
        return candidates
    kept_rows = []
    for _, trial_candidates in candidates.groupby("pipeline_trial_id", sort=False):
        kept = []
        for _, row in trial_candidates.sort_values("min_ms").iterrows():
            if not kept or row.min_ms - kept[-1].min_ms > config.MERGE_DISTANCE_MS:
                kept.append(row)
            elif row.prominence > kept[-1].prominence:
                kept[-1] = row
        kept_rows.extend(kept)
    return pd.DataFrame(kept_rows).sort_values(
        ["pipeline_trial_id", "onset_ms"]
    ).reset_index(drop=True)


def generate_candidates(frame_features):
    """Generate the exact 22-feature candidate table used by the RF."""
    rows = []
    for pipeline_trial_id, trial in frame_features.groupby(
        "pipeline_trial_id", sort=False
    ):
        rows.extend(_detect_trial_candidates(trial, pipeline_trial_id))
    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    return _merge_close_candidates(candidates)


def detect_blinks(frame_features, model_path):
    """Generate candidates, score them, and return accepted blink events."""
    candidates = generate_candidates(frame_features)
    event_columns = [
        "video_file",
        "participant_id",
        "trial_id",
        "pipeline_trial_id",
        "blink_onset_s",
        "blink_offset_s",
        "blink_midpoint_s",
        "rf_score",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=event_columns), candidates

    model = joblib.load(model_path)
    trained_features = list(getattr(model, "feature_names_in_", []))
    if trained_features != FEATURE_COLUMNS:
        raise ValueError("Random Forest artifact does not match the frozen feature order")

    candidates["rf_score"] = model.predict_proba(candidates[FEATURE_COLUMNS])[:, 1]
    accepted = candidates[candidates.rf_score >= config.RF_THRESHOLD].copy()
    trial_lookup = frame_features[
        ["pipeline_trial_id", "video_file", "participant_id", "trial_id"]
    ].drop_duplicates("pipeline_trial_id")
    accepted = accepted.merge(
        trial_lookup, on="pipeline_trial_id", how="left", validate="many_to_one"
    )
    accepted["blink_onset_s"] = accepted.onset_ms / 1000
    accepted["blink_offset_s"] = accepted.end_ms / 1000
    accepted["blink_midpoint_s"] = (
        accepted.blink_onset_s + accepted.blink_offset_s
    ) / 2
    return accepted[event_columns], candidates


if __name__ == "__main__":
    raise SystemExit("Run python run_pipeline.py from the deployment folder.")

