# src/evaluate.py
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test, class_names, out_dir: str = "results/quick_test"):
    os.makedirs(out_dir, exist_ok=True)
    y_pred = model.predict(X_test)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names, xticks_rotation="vertical", cmap="Blues"
    )
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(fig_path, dpi=180)
    plt.close()
    print(f"[Saved] {fig_path}")
