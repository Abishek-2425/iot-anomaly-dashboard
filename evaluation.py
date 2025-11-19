import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==== CONFIG ====
DATA_PATH = "synthetic_iot_dataset.csv"
MODEL_PATH = "model.pkl"

# ==== LOAD DATA AND MODEL ====
print("Loading model and dataset...")
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Detect numeric feature columns
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Anomaly' in feature_cols:
    feature_cols.remove('Anomaly')

X = df[feature_cols]
y_true = df['Anomaly'] if 'Anomaly' in df.columns else None

# ==== MAKE PREDICTIONS ====
print("Making predictions...")
y_pred = model.predict(X.values)

# IsolationForest outputs: -1 for anomaly, 1 for normal
# Convert to 1 (anomaly) and 0 (normal) to match dataset
y_pred_binary = np.where(y_pred == -1, 1, 0)

# ==== EVALUATION ====
if y_true is not None:
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
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Pred:Normal", "Pred:Anomaly"],
                yticklabels=["True:Normal", "True:Anomaly"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("\nConfusion matrix saved as 'confusion_matrix.png'.")
else:
    print("No 'Anomaly' column found in dataset. Evaluation skipped.")
