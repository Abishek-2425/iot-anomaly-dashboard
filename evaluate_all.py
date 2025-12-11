# evaluate_all.py
"""
Evaluate all three anomaly detection models using:
- Isolation Forest
- One-Class SVM
- Autoencoder

Outputs a *single* comparison table for all metrics.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

BASE = Path(__file__).parent
DATA_FILE = BASE / "synthetic_iot_dataset.csv"
OUTPUT_FILE = BASE / "output/confusion_matrices_all.png"
(BASE / "output").mkdir(exist_ok=True)



# ============================================================
#      CLEAN COMPARISON TABLE FORMATTER
# ============================================================

def print_comparison_table(results, sort_by="f1"):
    best_acc = max(r["accuracy"] for r in results)
    best_prec = max(r["precision"] for r in results)
    best_rec = max(r["recall"] for r in results)
    best_f1 = max(r["f1"] for r in results)

    results_sorted = sorted(results, key=lambda r: r[sort_by], reverse=True)

    line = "━" * 78
    print("\n" + Fore.MAGENTA + line + Style.RESET_ALL)
    print(Fore.CYAN + f"MODEL COMPARISON — SORTED BY {sort_by.upper()}".center(78) + Style.RESET_ALL)
    print(Fore.MAGENTA + line + Style.RESET_ALL)

    # Fixed widths
    col_model = 22
    col_num = 12

    header = (
        f"{'Model':<{col_model}} "
        f"{'Accuracy':<{col_num}} "
        f"{'Precision':<{col_num}} "
        f"{'Recall':<{col_num}} "
        f"{'F1-Score':<{col_num}}"
    )
    print(Fore.YELLOW + header + Style.RESET_ALL)
    print("-" * 78)

    for r in results_sorted:

        def fmt(val, best, width):
            raw = f"{val:.4f}"
            padded = f"{raw:<{width}}"  # pad BEFORE adding color
            if val == best:
                return Fore.GREEN + padded + Style.RESET_ALL
            return padded

        acc = fmt(r['accuracy'], best_acc, col_num)
        prec = fmt(r['precision'], best_prec, col_num)
        rec = fmt(r['recall'], best_rec, col_num)
        f1 = fmt(r['f1'], best_f1, col_num)

        print(
            f"{r['name']:<{col_model}} "
            f"{acc:<{col_num}} "
            f"{prec:<{col_num}} "
            f"{rec:<{col_num}} "
            f"{f1:<{col_num}}"
        )

    print("-" * 78 + "\n")

# ============================================================
#                 LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    print("Dataset not found:", DATA_FILE)
    raise SystemExit(1)

df = pd.read_csv(DATA_FILE)

if "Anomaly" not in df.columns:
    print("No 'Anomaly' column found. Evaluation aborted.")
    raise SystemExit(1)

y_true = df["Anomaly"].values

# Workaround inputs:
# Unsupervised models trained using 4 features → feed all 4.
# Autoencoder trained on 3 features → feed only 3.
X_if = df[["Temperature", "Humidity", "Battery_Level", "Anomaly"]].values
X_ocsvm = X_if
X_ae = df[["Temperature", "Humidity", "Battery_Level", "Anomaly"]].values


# ============================================================
#                MODEL CONFIG
# ============================================================
models_info = [
    {"name": "Isolation Forest", "file": BASE / "models/if_model.pkl", "type": "if"},
    {"name": "One-Class SVM", "file": BASE / "models/ocsvm_model.pkl", "type": "ocsvm"},
    {"name": "Autoencoder", "file": BASE / "models/autoencoder_model.keras", "type": "ae"},
]

scaler_ae = StandardScaler()
scaler_ocsvm = StandardScaler()


# ============================================================
#                PLOT SETUP
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))


# ============================================================
#                EVALUATE ALL MODELS
# ============================================================
results = []

for idx, info in enumerate(models_info):

    print(Fore.CYAN + f"\nEvaluating {info['name']}..." + Style.RESET_ALL)

    if info["type"] == "ae":
        model = load_model(info["file"])
        X_scaled = scaler_ae.fit_transform(X_ae)

        recon = model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - recon, 2), axis=1)
        threshold = np.percentile(mse, 95)
        y_pred = (mse > threshold).astype(int)

    elif info["type"] == "ocsvm":
        model = joblib.load(info["file"])
        X_scaled = scaler_ocsvm.fit_transform(X_ocsvm)

        raw = model.predict(X_scaled)
        y_pred = np.where(raw == -1, 1, 0)

    else:
        model = joblib.load(info["file"])
        raw = model.predict(X_if)
        y_pred = np.where(raw == -1, 1, 0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results.append({
        "name": info["name"],
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    })

    cm = confusion_matrix(y_true, y_pred)
    ax = axes[idx]
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{info['name']} CM")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")


# ============================================================
#        PRINT SINGLE COMPARISON TABLE
# ============================================================

print_comparison_table(results)

plt.tight_layout()
plt.savefig(OUTPUT_FILE)
plt.show()

print(Fore.GREEN + f"\nAll confusion matrices saved to {OUTPUT_FILE}" + Style.RESET_ALL)
