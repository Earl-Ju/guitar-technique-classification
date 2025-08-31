# src/models.py
from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def build_svm_pipeline(C: float = 1.0, kernel: str = "rbf", gamma: str = "scale") -> Pipeline:
    """SVM + 标准化的流水线"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=C, kernel=kernel, gamma=gamma, probability=True)),
    ])
