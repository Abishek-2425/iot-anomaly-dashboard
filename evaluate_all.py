# evaluate_all.py
"""
Evaluate all three anomaly detection models using:
- Isolation Forest
- One-Class SVM
- Autoencoder

Outputs a *single* comparison table for all metrics.
"""

# ==== SUPPRESS TENSORFLOW LOGS (must be before importing TF) ====
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
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
def print_comparison_table(results):
    """Prints a clean single comparison table of all models."""
    line = "━" * 72

    print("\n" + Fore.MAGENTA + line + Style.RESET_ALL)
    print(Fore.CYAN + "MODEL COMPARISON (IF • OCSVM • AUTOENCODER)".center(72) + Style.RESET_ALL)
    print(Fore.MAGENTA + line + Style.RESET_ALL)

    header = (
        f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} "
        f"{'Recall':<10} {'F1-Score':<10}"
    )
    print(Fore.YELLOW + header + Style.RESET_ALL)
    print("-" * 72)

    for r in results:
        print(
            f"{r['name']:<20} "
            f"{r['accuracy']:<10.4f} "
            f"{r['precision']:<10.4f} "
            f"{r['recall']:<10.4f} "
            f"{r['f1']:<10.4f}"
        )

    print("-" * 72 + "\n")


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

# Features used for model evaluation
feature_cols = ["Temperature", "Humidity", "Battery_Level", "Anomaly"]
X_model = df[feature_cols].values


# ============================================================
#                MODEL CONFIG
# ============================================================
models_info = [
    {"name": "Isolation Forest", "file": BASE / "models/if_model.pkl", "type": "if"},
    {"name": "One-Class SVM", "file": BASE / "models/ocsvm_model.pkl", "type": "ocsvm"},
    {"name": "Autoencoder", "file": BASE / "models/autoencoder_model.keras", "type": "ae"},
]

scaler = StandardScaler()


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

    # Load model and predict
    if info["type"] == "ae":
        model = load_model(info["file"])
        X_scaled = scaler.fit_transform(X_model)
        recon = model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - recon, 2), axis=1)
        threshold = np.percentile(mse, 95)
        y_pred = (mse > threshold).astype(int)

    else:
        model = joblib.load(info["file"])
        y_pred_raw = model.predict(X_model)
        y_pred = np.where(y_pred_raw == -1, 1, 0)

    # Compute metrics
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

    # Plot confusion matrix
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
