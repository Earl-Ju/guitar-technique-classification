# src/train.py
from __future__ import annotations
import argparse
from pathlib import Path

from .dataset import load_folder_classification
from .models import get_model, save_model
from .evaluate import evaluate_and_plot

def main():
    parser = argparse.ArgumentParser(description="Train SVM baseline on folder-structured dataset.")
    parser.add_argument("--data", type=str, default="data/quick_test", help="Root folder: class subfolders with audio.")
    parser.add_argument("--out", type=str, default="results/quick_test", help="Output dir for artifacts.")
    parser.add_argument("--model", type=str, default="svm", help="Model kind (svm).")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔹 Loading dataset from: {args.data}")
    X_train, X_test, y_train, y_test, class_names = load_folder_classification(args.data)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}, Classes: {class_names}")

    print("🔹 Building model pipeline…")
    model = get_model(args.model)

    print("🔹 Training…")
    model.fit(X_train, y_train)

    print("🔹 Evaluating…")
    fig = evaluate_and_plot(model, X_test, y_test, class_names)
    fig_path = out_dir / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=160)
    print(f"   Saved: {fig_path}")

    model_path = out_dir / f"{args.model}_baseline.joblib"
    save_model(model, model_path)
    print(f"   Saved model: {model_path}")

    print("✅ Done.")

if __name__ == "__main__":
    main()
