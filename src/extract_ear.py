"""MediaPipe landmark extraction and the validated caregiver EAR calculation."""

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(eye_points):
    """Calculate EAR from six ordered pixel-coordinate eye landmarks."""
    vertical_1 = math.dist(eye_points[1], eye_points[5])
    vertical_2 = math.dist(eye_points[2], eye_points[4])
    horizontal = math.dist(eye_points[0], eye_points[3])
    if horizontal == 0:
        return np.nan
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def landmark_to_pixel(landmark, width, height):
    """Preserve the historical truncation to integer pixel coordinates."""
    return int(landmark.x * width), int(landmark.y * height)


def create_face_landmarker(model_path):
    """Create the historical one-face MediaPipe detector in video mode."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def extract_ear_from_frame(mp, detector, frame, frame_index, video_fps):
    """Extract bilateral EAR and MediaPipe blink scores from one frame."""
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detector_time_ms = int(frame_index / video_fps * 1000)
    result = detector.detect_for_video(mp_image, detector_time_ms)
    timestamp_ms = frame_index / video_fps * 1000

    if not result.face_landmarks:
        return {
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "left_ear": np.nan,
            "right_ear": np.nan,
            "ear": np.nan,
            "blink_left": np.nan,
            "blink_right": np.nan,
            "blink_score": np.nan,
            "tracking_valid": False,
        }

    face = result.face_landmarks[0]
    left_points = [landmark_to_pixel(face[index], width, height) for index in LEFT_EYE]
    right_points = [
        landmark_to_pixel(face[index], width, height) for index in RIGHT_EYE
    ]
    left_ear = eye_aspect_ratio(left_points)
    right_ear = eye_aspect_ratio(right_points)

    blink_left = np.nan
    blink_right = np.nan
    if result.face_blendshapes:
        blendshapes = result.face_blendshapes[0]
        scores = {item.category_name: item.score for item in blendshapes}
        if "eyeBlinkLeft" in scores and "eyeBlinkRight" in scores:
            blink_left = scores["eyeBlinkLeft"]
            blink_right = scores["eyeBlinkRight"]
        elif len(blendshapes) > 10:
            blink_left = blendshapes[9].score
            blink_right = blendshapes[10].score

    return {
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "left_ear": left_ear,
        "right_ear": right_ear,
        "ear": (left_ear + right_ear) / 2.0,
        "blink_left": blink_left,
        "blink_right": blink_right,
        "blink_score": (blink_left + blink_right) / 2.0,
        "tracking_valid": True,
    }


def _check_non_overlapping_trials(video_trials):
    ordered = video_trials.sort_values(["onset_s", "offset_s"])
    previous_end = None
    for trial in ordered.itertuples(index=False):
        if previous_end is not None and trial.onset_s <= previous_end:
            raise ValueError(
                f"Trials for {trial.video_file} overlap or share an endpoint. "
                "The validated MediaPipe video-mode extractor requires strictly "
                "increasing, non-overlapping intervals."
            )
        previous_end = trial.offset_s
    return ordered


def extract_video_ear(video_path, video_trials, output_path, model_path):
    """Extract the requested intervals and save one per-video EAR cache."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    trials = _check_non_overlapping_trials(video_trials)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or total_frames <= 0:
        capture.release()
        raise ValueError(f"Could not read video timing: {video_path}")

    mp, detector = create_face_landmarker(model_path)
    rows = []
    try:
        for trial in trials.itertuples(index=False):
            start_frame = max(1, int(trial.onset_s * fps))
            end_frame = min(total_frames, int(trial.offset_s * fps))
            if end_frame < start_frame:
                continue

            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
            frame_index = start_frame - 1
            frame_count = end_frame - start_frame + 1
            description = f"{video_path.name} trial {trial.trial_id}"
            with tqdm(total=frame_count, desc=description, unit="frame") as progress:
                while frame_index < end_frame:
                    success, frame = capture.read()
                    if not success:
                        break
                    frame_index += 1
                    row = extract_ear_from_frame(
                        mp, detector, frame, frame_index, fps
                    )
                    row.update(
                        {
                            "video_file": trial.video_file,
                            "participant_id": trial.participant_id,
                            "trial_id": trial.trial_id,
                            "pipeline_trial_id": trial.pipeline_trial_id,
                            "trial_onset_ms": trial.onset_s * 1000,
                            "trial_offset_ms": trial.offset_s * 1000,
                        }
                    )
                    rows.append(row)
                    progress.update(1)
    finally:
        detector.close()
        capture.release()

    features = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return features
