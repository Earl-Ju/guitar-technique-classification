# src/models.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def get_model(kind: Literal["svm"]="svm") -> Pipeline:
    """
    返回一个可直接用于训练/预测的 sklearn Pipeline。
    这里给出 SVM 基线：StandardScaler + RBF-SVM
    """
    if kind == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=True)),
        ])
    raise ValueError(f"Unknown model kind: {kind}")

def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str | Path):
    return joblib.load(path)
