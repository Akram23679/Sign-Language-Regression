#!/usr/bin/env python3
"""
collect_data.py
===============
Record hand-landmark sequences from your webcam to train a dynamic
sign-language recognizer.

It uses Google's MediaPipe **Hand Landmarker (Tasks API)** -- the SAME model the
Flutter `hand_landmarker` plugin uses on-device -- so the features you record
here match what the phone will produce at inference time. Keeping that identical
is the single most important thing for this whole project to work.

------------------------------------------------------------------------------
WHAT IT SAVES
------------------------------------------------------------------------------
For every sign, it records SEQUENCES_PER_SIGN takes. Each take is
FRAMES_PER_SEQUENCE frames. Each frame is a fixed-length vector of 126 floats:

    [ LEFT hand  : 21 landmarks x (x, y, z) = 63 values ]   indices  0..62
    [ RIGHT hand : 21 landmarks x (x, y, z) = 63 values ]   indices 63..125

  - Hand slot is by wrist x-position: left-most hand in the image -> slot 0.
    (We avoid left/right "handedness" because the Flutter package can't report it.)
  - A missing hand is all zeros.
  - x, y, z are MediaPipe's normalized coordinates (raw -- NOT yet wrist-centered).
    Normalization happens later in preprocess.py, and must be mirrored identically
    on the Flutter side.
  - The preview is NOT mirrored, so it matches a phone's raw front-camera frames.
    (Your movements will look "reversed" on screen -- that's expected.)

Files are written as:  data/<sign>/<index>.npy   with shape (FRAMES_PER_SEQUENCE, 126)

------------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------------
    pip install opencv-python mediapipe numpy

Then just run it:
    python collect_data.py

The MediaPipe model file (hand_landmarker.task) auto-downloads on first run.

------------------------------------------------------------------------------
CONTROLS (during the live window)
------------------------------------------------------------------------------
    SPACE  start recording the next take
    s      skip / redo the current take (don't save)
    n      jump to the next sign
    q      quit (progress is saved per-take, so you can resume anytime)
"""

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ----------------------------- CONFIG ---------------------------------------
SIGNS = ["hello", "thanks", "yes", "no", "please", "sorry", "help", "iloveyou"]
SEQUENCES_PER_SIGN = 40        # takes per sign (more + more varied = better)
FRAMES_PER_SEQUENCE = 30       # ~1 second per take
CAM_INDEX = 0                  # change if you have multiple cameras

NUM_HANDS = 2
LM_PER_HAND = 21
FEATURES_PER_FRAME = NUM_HANDS * LM_PER_HAND * 3   # = 126

DATA_DIR = Path("data")
MODEL_PATH = Path("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
# ----------------------------------------------------------------------------

# Standard MediaPipe 21-point hand topology (hardcoded so we don't depend on
# the legacy mp.solutions API, which is broken/removed in recent mediapipe).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                               # palm base
]


def ensure_model():
    """Download the MediaPipe hand-landmarker model on first run."""
    if MODEL_PATH.exists():
        return
    print(f"Downloading hand landmarker model -> {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


def make_detector():
    base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=NUM_HANDS,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_features(result) -> np.ndarray:
    """Fixed 126-length vector. Hands are ordered by wrist x-position
    (left-most in the image -> slot 0). We do NOT use MediaPipe handedness,
    because the Flutter `hand_landmarker` package does not expose it -- so the
    phone must order hands the exact same way. Keep this identical on both sides.
    """
    feats = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    if not result.hand_landmarks:
        return feats
    hands = sorted(result.hand_landmarks, key=lambda lms: lms[0].x)  # by wrist x
    for slot, lms in enumerate(hands[:2]):
        base = slot * LM_PER_HAND * 3
        for i, lm in enumerate(lms):
            feats[base + 3 * i: base + 3 * i + 3] = (lm.x, lm.y, lm.z)
    return feats


def draw_landmarks(frame, result):
    h, w = frame.shape[:2]
    for lms in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 180, 0), 2)
        for (x, y) in pts:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def existing_takes(sign: str) -> int:
    folder = DATA_DIR / sign
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.npy")))


def banner(frame, lines):
    """Draw a few lines of text with a dark strip behind them."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 30 + 26 * len(lines)),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (12, 28 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def main():
    ensure_model()
    DATA_DIR.mkdir(exist_ok=True)
    detector = make_detector()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {CAM_INDEX}.")

    ts = 0  # monotonically increasing timestamp (ms) required by VIDEO mode

    for sign in SIGNS:
        (DATA_DIR / sign).mkdir(exist_ok=True)
        take = existing_takes(sign)            # resume where we left off
        if take >= SEQUENCES_PER_SIGN:
            continue

        skip_sign = False
        while take < SEQUENCES_PER_SIGN and not skip_sign:
            # ---- IDLE: wait for SPACE to start the next take ----
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit("Camera read failed.")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts += 33
                result = detector.detect_for_video(mp_img, ts)
                draw_landmarks(frame, result)

                banner(frame, [
                    (f"SIGN: {sign}   take {take + 1}/{SEQUENCES_PER_SIGN}",
                     (255, 255, 255)),
                    ("SPACE=record   s=skip   n=next sign   q=quit",
                     (180, 220, 180)),
                ])
                cv2.imshow("collect_data", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return
                if key == ord("n"):
                    skip_sign = True; break
                if key == ord(" "):
                    break
            if skip_sign:
                break

            # ---- COUNTDOWN: 3..2..1 while still showing the live feed ----
            t0 = time.time()
            while time.time() - t0 < 1.5:
                ok, frame = cap.read()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts += 33
                result = detector.detect_for_video(mp_img, ts)
                draw_landmarks(frame, result)
                left = 1.5 - (time.time() - t0)
                banner(frame, [(f"GET READY... {left:0.1f}", (0, 215, 255))])
                cv2.imshow("collect_data", frame)
                cv2.waitKey(1)

            # ---- RECORD: capture FRAMES_PER_SEQUENCE frames ----
            sequence = []
            aborted = False
            while len(sequence) < FRAMES_PER_SEQUENCE:
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit("Camera read failed.")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts += 33
                result = detector.detect_for_video(mp_img, ts)
                sequence.append(extract_features(result))
                draw_landmarks(frame, result)

                banner(frame, [
                    (f"RECORDING {sign}  frame {len(sequence)}/{FRAMES_PER_SEQUENCE}",
                     (0, 0, 255)),
                    ("s = abort this take", (180, 180, 180)),
                ])
                cv2.imshow("collect_data", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return
                if key == ord("s"):
                    aborted = True; break

            if aborted:
                continue   # redo this take without saving

            out = DATA_DIR / sign / f"{take:03d}.npy"
            np.save(out, np.array(sequence, dtype=np.float32))
            print(f"saved {out}  shape={np.array(sequence).shape}")
            take += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nAll done. Next step: preprocess.py to normalize + build the dataset.")


if __name__ == "__main__":
    main()