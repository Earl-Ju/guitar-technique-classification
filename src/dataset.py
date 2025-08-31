# src/dataset.py
from __future__ import annotations
import os, json, glob
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from .features import extract_features

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")

def scan_class_folders(root: str) -> Dict[str, List[str]]:
    classes = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        files = []
        for ext in AUDIO_EXTS:
            files += glob.glob(os.path.join(p, f"*{ext}"))
        if files:
            classes[name] = sorted(files)
    if not classes:
        raise FileNotFoundError(f"No class folders with audio under: {root}")
    return classes

def build_label_map(class_names: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(sorted(class_names))}

def load_folder_classification(
    root: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return X_train, X_test, y_train, y_test, class_names."""
    class_files = scan_class_folders(root)
    class_names = sorted(class_files.keys())
    label_map = build_label_map(class_names)

    X, y = [], []
    for label, files in class_files.items():
        for f in files:
            try:
                feat = extract_features(f)
                X.append(feat)
                y.append(label_map[label])
            except Exception as e:
                print(f"[WARN] skip {f}: {e}")

    if len(X) == 0:
        raise RuntimeError("No valid audio features extracted.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if len(np.unique(y)) < 2:
        # 无法分层切分时退化为普通切分
        return *train_test_split(X, y, test_size=test_size, random_state=random_state), class_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, class_names

def save_label_map(label_map_path: str, class_names: List[str]) -> None:
    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({i: n for i, n in enumerate(class_names)}, f, ensure_ascii=False, indent=2)
