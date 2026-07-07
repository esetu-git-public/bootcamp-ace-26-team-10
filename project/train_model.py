
# train_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone script to train, evaluate, and save the best CKD classifier.
# Run:  python train_model.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics          import (accuracy_score, precision_score, recall_score,
                                      f1_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(BASE_DIR, "dataset", "Training_CKD_dataset.csv")
TEST_PATH   = os.path.join(BASE_DIR, "dataset", "Testing_CKD_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "model",   "kidney_model.pkl")
ASSETS_DIR  = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ─── Add project root to sys.path ────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from ckd_utils.preprocessing import preprocess_pipeline, check_data_quality, load_data


def train_and_evaluate():
    print("=" * 70)
    print("  Chronic Kidney Disease - Model Training Pipeline")
    print("=" * 70)

    # ── 1. Load & Quality Check ───────────────────────────────────────────
    print("\n[1/5] Loading datasets ...")
    train_raw = load_data(TRAIN_PATH)
    test_raw  = load_data(TEST_PATH)

    q = check_data_quality(train_raw)
    print(f"  Training set  : {q['shape'][0]:,} rows × {q['shape'][1]} cols")
    print(f"  Missing values: {q['total_missing']}")
    print(f"  Duplicate rows: {q['duplicate_rows']}")

    # ── 2. Full Preprocessing ─────────────────────────────────────────────
    print("\n[2/5] Preprocessing ...")
    X_train, X_test, y_train, y_test, encoders, scaler, class_names = \
        preprocess_pipeline(TRAIN_PATH, TEST_PATH)
    print("\n===== FEATURE CHECK =====")
    print("Target column in X_train:", "Target" in X_train.columns)
    print("Number of features:", len(X_train.columns))
    print("Feature names:")
    print(list(X_train.columns))
    print("=========================\n")

    print(f"  Features : {X_train.shape[1]}")
    print(f"  Classes  : {class_names}")

    # ── 3. Define Models ──────────────────────────────────────────────────
    print("\n[3/5] Training classifiers ...")
    models = {
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
        "Decision Tree":        DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results      = {}
    best_name    = None
    best_f1      = -1
    best_model   = None

    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1_score":  round(f1,   4),
            "model":     clf,
            "y_pred":    y_pred,
            "time_s":    round(time.time() - t0, 2),
        }
        print(f"  {name:<25}  Acc={acc:.4f}  F1={f1:.4f}  ({results[name]['time_s']}s)")

        if f1 > best_f1:
            best_f1    = f1
            best_name  = name
            best_model = clf

    # ── 4. Save Best Model Bundle ─────────────────────────────────────────
    print(f"\n[4/5] Best model -> {best_name}  (F1={best_f1:.4f})")

    # Build comparison metrics (exclude model/y_pred objects)
    comparison_metrics = {
        name: {k: v for k, v in res.items() if k not in ("model", "y_pred")}
        for name, res in results.items()
    }

    # Full classification report for the best model
    best_y_pred = results[best_name]["y_pred"]
    report_str  = classification_report(y_test, best_y_pred,
                                        target_names=class_names,
                                        zero_division=0)
    cm = confusion_matrix(y_test, best_y_pred)

    bundle = {
        "model":              best_model,
        "model_name":         best_name,
        "encoders":           encoders,
        "scaler":             scaler,
        "classes":            class_names,
        "features":           list(X_train.columns),
        
        
    }

    joblib.dump(bundle, MODEL_PATH)
    
    joblib.load(MODEL_PATH)
    print("Bundle loaded successfully!")

    print(f"  Saved → {MODEL_PATH}")

    # ── 5. Save Assets ────────────────────────────────────────────────────
    print("\n[5/5] Generating plots …")

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(f"Confusion Matrix – {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(ASSETS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {cm_path}")

    # Model Comparison Bar Chart
    metric_df = pd.DataFrame(
        {n: {k: v for k, v in r.items() if k in ("accuracy","precision","recall","f1_score")}
         for n, r in results.items()}
    ).T
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_df.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white", width=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    comp_path = os.path.join(ASSETS_DIR, "model_comparison.png")
    fig.savefig(comp_path, dpi=150)
    plt.close(fig)
    print(f"  Model comparison → {comp_path}")

    print("\n" + "=" * 70)
    print("  Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
