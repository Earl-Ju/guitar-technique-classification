# src/dataset.py
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

from .features import extract_features

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

def iter_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS and p.is_file():
            files.append(p)
    return sorted(files)

def load_folder_classification(
    data_root: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    假设目录结构：
        data_root/
          classA/*.wav
          classB/*.wav
          ...
    返回：X_train, X_test, y_train, y_test, class_names
    """
    data_root = Path(data_root)
    class_dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    class_names = [d.name for d in class_dirs]

    X, y = [], []
    for label, d in enumerate(class_dirs):
        for f in iter_audio_files(d):
            try:
                feats = extract_features(str(f))
                X.append(feats)
                y.append(label)
            except Exception as e:
                print(f"[WARN] skip {f}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None
    )
    return X_train, X_test, y_train, y_test, class_names
