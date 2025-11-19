# eval_if.py
"""
Evaluate trained Isolation Forest model on synthetic_iot_dataset.csv.
Generates confusion matrix, classification report, and evaluation metrics.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE = Path(__file__).parent
DATA_FILE = BASE / "synthetic_iot_dataset.csv"
MODEL_FILE = BASE / "models/base_if_model.pkl"
OUTPUT_FILE = BASE / "outputs/isolationforest.png"

# ==== LOAD DATA AND MODEL ====
if not DATA_FILE.exists():
    print("Dataset not found at", DATA_FILE)
    raise SystemExit(1)

print("Loading model and dataset...")
model = joblib.load(MODEL_FILE)
df = pd.read_csv(DATA_FILE)

# Select numeric features
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "Anomaly" in feature_cols:
    feature_cols.remove("Anomaly")

X = df[feature_cols].values
if "Anomaly" in df.columns:
    y_true = df["Anomaly"].values
else:
    print("No 'Anomaly' column found. Evaluation skipped.")
    raise SystemExit(1)

# ==== MAKE PREDICTIONS ====
print("Making predictions...")
y_pred = model.predict(X)
# IsolationForest outputs: 1 = normal, -1 = anomaly
y_pred_binary = np.where(y_pred == -1, 1, 0)  # 1 = anomaly, 0 = normal

# ==== EVALUATION METRICS ====
print("\n--- Evaluation Metrics ---")
acc = accuracy_score(y_true, y_pred_binary)
prec = precision_score(y_true, y_pred_binary, zero_division=0)
rec = recall_score(y_true, y_pred_binary, zero_division=0)
f1 = f1_score(y_true, y_pred_binary, zero_division=0)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred_binary, target_names=["Normal", "Anomaly"]))

# ==== CONFUSION MATRIX ====
cm = confusion_matrix(y_true, y_pred_binary)
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')
plt.title("Isolation Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Normal", "Anomaly"])
plt.yticks([0,1], ["Normal", "Anomaly"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='red')

plt.tight_layout()
plt.savefig(OUTPUT_FILE)
plt.close()
print("\nConfusion matrix saved to", OUTPUT_FILE)
