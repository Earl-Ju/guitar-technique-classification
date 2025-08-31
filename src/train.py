# src/train.py
from __future__ import annotations
import os, argparse, joblib
from .dataset import load_folder_classification, save_label_map
from .models import build_svm_pipeline
from .evaluate import evaluate_model

def parse_args():
    p = argparse.ArgumentParser(description="Train SVM baseline for guitar technique classification")
    p.add_argument("--data_dir", default="data/processed", help="Root folder with class subfolders")
    p.add_argument("--out_dir",  default="results/quick_test", help="Where to save artifacts")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load data/features
    X_train, X_test, y_train, y_test, class_names = load_folder_classification(
        args.data_dir, test_size=args.test_size, random_state=args.seed
    )

    # 2) Build model & train
    model = build_svm_pipeline()
    model.fit(X_train, y_train)

    # 3) Evaluate
    evaluate_model(model, X_test, y_test, class_names, out_dir=args.out_dir)

    # 4) Save model + label map
    model_path = os.path.join(args.out_dir, "svm_baseline.joblib")
    joblib.dump(model, model_path)
    save_label_map(os.path.join(args.out_dir, "label_map.json"), class_names)

    print(f"[Saved] {model_path}")

if __name__ == "__main__":
    main()
