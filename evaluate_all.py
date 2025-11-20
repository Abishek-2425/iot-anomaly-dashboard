# evaluate_all.py
"""
Evaluate all three anomaly detection models using the exact features used for training:
- Isolation Forest
- One-Class SVM
- Autoencoder

Uses synthetic_iot_dataset.csv at project root and models in models/.
Generates metrics, classification reports, and confusion matrices.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

BASE = Path(__file__).parent
DATA_FILE = BASE / "synthetic_iot_dataset.csv"
OUTPUT_FILE = BASE / "output/confusion_matrices_all.png"
(BASE / "output").mkdir(exist_ok=True)

# ==== LOAD DATA ====
if not DATA_FILE.exists():
    print("Dataset not found at", DATA_FILE)
    raise SystemExit(1)

df = pd.read_csv(DATA_FILE)
if "Anomaly" not in df.columns:
    print("No 'Anomaly' column found. Evaluation skipped.")
    raise SystemExit(1)

y_true = df["Anomaly"].values

# ==== FIXED FEATURES ====
feature_cols = ["Temperature", "Humidity", "Battery_Level","Anomaly"]
X_model = df[feature_cols].values

# ==== MODELS CONFIG ====
models_info = [
    {"name": "Isolation Forest", "file": BASE / "models/if_model.pkl", "type": "if"},
    {"name": "One-Class SVM", "file": BASE / "models/ocsvm_model.pkl", "type": "ocsvm"},
    {"name": "Autoencoder", "file": BASE / "models/autoencoder_model.keras", "type": "ae"}
]

scaler = StandardScaler()  # Only for Autoencoder

# ==== PLOT SETUP ====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, info in enumerate(models_info):
    print(f"\nEvaluating {info['name']}...")

    # Load model
    if info["type"] == "ae":
        model = load_model(info["file"])
        X_scaled = scaler.fit_transform(X_model)
        reconstructions = model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        threshold = np.percentile(mse, 95)  # 95th percentile threshold
        y_pred = (mse > threshold).astype(int)
    else:
        model = joblib.load(info["file"])
        y_pred_raw = model.predict(X_model)
        y_pred = np.where(y_pred_raw == -1, 1, 0)  # anomaly = 1, normal = 0

    # ==== METRICS ====
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

    # ==== CONFUSION MATRIX ====
    cm = confusion_matrix(y_true, y_pred)
    ax = axes[idx]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(f"{info['name']} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

plt.tight_layout()
plt.savefig(OUTPUT_FILE)
plt.show()
print(f"\nAll confusion matrices saved to {OUTPUT_FILE}")
