# evaluate_all.py
"""
Evaluate all three anomaly detection models using the exact features used for training:
- Isolation Forest
- One-Class SVM
- Autoencoder
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

# Initialize Colorama for Windows
colorama_init(autoreset=True)

BASE = Path(__file__).parent
DATA_FILE = BASE / "synthetic_iot_dataset.csv"
OUTPUT_FILE = BASE / "output/confusion_matrices_all.png"
(BASE / "output").mkdir(exist_ok=True)


# ============================================================
#  CLEAN COLORED OUTPUT FORMATTER
# ============================================================
def print_block(model_name, accuracy, precision, recall, f1, report_dict):
    line = "━" * 60

    print("\n" + Fore.MAGENTA + line + Style.RESET_ALL)
    print(Fore.CYAN + model_name.center(60) + Style.RESET_ALL)
    print(Fore.MAGENTA + line + Style.RESET_ALL)

    print(Fore.YELLOW + f"Accuracy : {accuracy:.4f}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"Precision: {precision:.4f}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"Recall   : {recall:.4f}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"F1 Score : {f1:.4f}\n" + Style.RESET_ALL)

    print(Fore.GREEN + "Class Breakdown:" + Style.RESET_ALL)
    for cls, stats in report_dict.items():
        if cls in ["accuracy", "macro avg", "weighted avg"]:
            continue

        p = stats["precision"]
        r = stats["recall"]
        f = stats["f1-score"]

        print(
            Fore.WHITE +
            f"  {cls:<7} → Precision {p:.2f} | Recall {r:.2f} | F1 {f:.2f}"
            + Style.RESET_ALL
        )

    print(Fore.MAGENTA + line + Style.RESET_ALL + "\n")


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

scaler = StandardScaler()  # For Autoencoder

# ============================================================
#                PLOT SETUP
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))


# ============================================================
#                EVALUATE ALL MODELS
# ============================================================
for idx, info in enumerate(models_info):

    # Colored model heading
    print(Fore.CYAN + f"\nEvaluating {info['name']}..." + Style.RESET_ALL)

    # Load correct model type
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

    # ---------- METRICS ----------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Anomaly"],
        output_dict=True
    )

    # Use cleaner output block
    print_block(info["name"], acc, prec, rec, f1, report)

    # ---------- CONFUSION MATRIX ----------
    cm = confusion_matrix(y_true, y_pred)
    ax = axes[idx]
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{info['name']} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")


plt.tight_layout()
plt.savefig(OUTPUT_FILE)
plt.show()

print(Fore.GREEN + f"\nAll confusion matrices saved to {OUTPUT_FILE}" + Style.RESET_ALL)
