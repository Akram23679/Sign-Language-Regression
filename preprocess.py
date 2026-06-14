#!/usr/bin/env python3
"""
preprocess.py
=============
Turn the raw landmark takes recorded by collect_data.py into a clean, normalized
dataset ready for training.

It does three things:
  1. Loads every data/<sign>/*.npy  (each shape: FRAMES x 126)
  2. NORMALIZES each frame so the model generalizes across where the hand is in
     the frame and how far it is from the camera.
  3. Splits into train/test and saves:
        dataset.npz   ->  X_train, X_test, y_train, y_test
        labels.json   ->  ["hello", "thanks", ...]  (index = class id)

------------------------------------------------------------------------------
NORMALIZATION  (must be mirrored EXACTLY on the Flutter side -- this is the
                whole reason the project works on the phone)
------------------------------------------------------------------------------
The 126 values per frame are two hands, each 21 landmarks x (x, y, z):
    LEFT  hand -> indices  0..62
    RIGHT hand -> indices 63..125
For EACH hand that is present (a missing hand is all zeros and stays zeros):
    a) translate: subtract the wrist (landmark 0) from all 21 landmarks
       -> the hand is now centered on its own wrist (position-invariant)
    b) scale: divide every coordinate by the largest distance from the wrist
       to any landmark of that hand
       -> the biggest spread becomes 1.0 (size / camera-distance-invariant)
An absent hand (all zeros) is left as zeros.

That's it -- translate to wrist, scale by max reach. Easy to reproduce in Dart.

------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------
    python preprocess.py
"""

import json
from pathlib import Path

import numpy as np

# ---- must match collect_data.py ----
FRAMES_PER_SEQUENCE = 30
FEATURES_PER_FRAME = 126        # 2 hands x 21 landmarks x 3
LM_PER_HAND = 21
DATA_DIR = Path("data")
OUT_NPZ = Path("dataset.npz")
OUT_LABELS = Path("labels.json")
TEST_FRACTION = 0.2
SEED = 42
# ------------------------------------


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Wrist-center + scale each present hand. Mirror this exactly on Flutter."""
    out = frame.astype(np.float32).copy()
    for slot in range(2):                       # 0 = left, 1 = right
        base = slot * LM_PER_HAND * 3
        hand = out[base: base + LM_PER_HAND * 3].reshape(LM_PER_HAND, 3)
        if not np.any(hand):                    # absent hand -> keep zeros
            continue
        hand = hand - hand[0]                   # (a) translate to wrist
        reach = np.linalg.norm(hand, axis=1).max()
        if reach > 1e-6:                        # (b) scale by max reach
            hand = hand / reach
        out[base: base + LM_PER_HAND * 3] = hand.reshape(-1)
    return out


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    return np.stack([normalize_frame(f) for f in seq], axis=0)


def stratified_split(X, y, test_frac, seed):
    """Per-class shuffle + split, so every sign appears in both train and test."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = np.array(train_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    if not DATA_DIR.exists():
        raise SystemExit("No data/ folder found. Run collect_data.py first.")

    # signs = folders that actually contain takes, in a stable (sorted) order
    signs = sorted(d.name for d in DATA_DIR.iterdir()
                   if d.is_dir() and any(d.glob("*.npy")))
    if not signs:
        raise SystemExit("data/ has no recorded takes yet. Run collect_data.py.")

    X, y = [], []
    print("Loading + normalizing:")
    for label, sign in enumerate(signs):
        files = sorted((DATA_DIR / sign).glob("*.npy"))
        kept = 0
        for f in files:
            seq = np.load(f)
            if seq.shape != (FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME):
                print(f"  ! skipping {f} (unexpected shape {seq.shape})")
                continue
            X.append(normalize_sequence(seq))
            y.append(label)
            kept += 1
        flag = "  <-- few samples, record more" if kept < 5 else ""
        print(f"  [{label}] {sign:<10} {kept} takes{flag}")

    X = np.array(X, dtype=np.float32)           # (N, 30, 126)
    y = np.array(y, dtype=np.int64)             # (N,)
    print(f"\nTotal: {len(X)} sequences across {len(signs)} signs, X shape {X.shape}")

    X_train, X_test, y_train, y_test = stratified_split(X, y, TEST_FRACTION, SEED)
    print(f"Train: {len(X_train)}   Test: {len(X_test)}")

    np.savez_compressed(OUT_NPZ,
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test)
    OUT_LABELS.write_text(json.dumps(signs, indent=2))
    print(f"\nSaved {OUT_NPZ} and {OUT_LABELS}")
    print("Next step: train.py  (build the LSTM, evaluate, export TFLite)")


if __name__ == "__main__":
    main()