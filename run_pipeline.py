#!/usr/bin/env python3
"""Run the frozen caregiver blink-detection pipeline."""

from pathlib import Path
import sys

import pandas as pd

import config


sys.path.insert(0, str(config.BASE_DIR / "src"))

from detect_blinks import detect_blinks
from extract_ear import extract_video_ear
from io_utils import (
    cache_matches_trials,
    load_trials,
    processing_summary,
    resolve_video_paths,
)


def check_required_files():
    required = [config.TRIALS_FILE, config.FACE_MODEL_FILE, config.RF_MODEL_FILE]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required pipeline files are missing:\n"
            + "\n".join(f"- {path}" for path in missing)
        )


def load_or_extract(video_path, video_trials):
    cache_path = config.FEATURES_DIR / f"{Path(video_path).stem}_ear.csv"
    if config.REUSE_FEATURE_CACHE and cache_path.is_file():
        cached = pd.read_csv(cache_path)
        if cache_matches_trials(cached, video_trials):
            print(f"Using cached EAR: {cache_path.name}")
            return cached
        print(f"Trial sheet changed; rebuilding cache: {cache_path.name}")
    return extract_video_ear(
        video_path, video_trials, cache_path, config.FACE_MODEL_FILE
    )


def main():
    check_required_files()
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    trials = load_trials(config.TRIALS_FILE)
    trials, video_paths = resolve_video_paths(trials, config.VIDEOS_DIR)

    all_features = []
    all_events = []
    for video_file, video_trials in trials.groupby("video_file", sort=False):
        features = load_or_extract(video_paths[str(video_file)], video_trials)
        events, candidates = detect_blinks(features, config.RF_MODEL_FILE)
        print(
            f"{video_file}: {len(candidates)} EAR-dip candidates, "
            f"{len(events)} retained by the Random Forest"
        )
        all_features.append(features)
        all_events.append(events)

    features = pd.concat(all_features, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True)
    events = events.sort_values(
        ["video_file", "pipeline_trial_id", "blink_midpoint_s"]
    ).reset_index(drop=True)
    summary = processing_summary(trials, features, events)

    events[
        [
            "video_file",
            "participant_id",
            "trial_id",
            "blink_onset_s",
            "blink_offset_s",
            "blink_midpoint_s",
            "rf_score",
        ]
    ].to_csv(config.BLINK_RESULTS_FILE, index=False)
    summary.to_csv(config.PROCESSING_SUMMARY_FILE, index=False)

    print()
    print(f"Saved {config.BLINK_RESULTS_FILE}")
    print(f"Saved {config.PROCESSING_SUMMARY_FILE}")


if __name__ == "__main__":
    main()

