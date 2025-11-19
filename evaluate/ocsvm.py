# eval_ocsvm.py
"""
Evaluate trained One-Class SVM on test data.
Generates confusion matrix and classification report.
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
DATA_FILE = BASE / "synthetic_iot_dataset.csv"
MODEL_FILE = BASE / "models/ocsvm_model.pkl"
OUTPUT_FILE = BASE / "outputs/ocsvm.png"

if not DATA_FILE.exists():
    print("Dataset not found at", DATA_FILE)
    raise SystemExit(1)

df = pd.read_csv(DATA_FILE)
numeric = df.select_dtypes(include=[int, float]).drop(columns=["Anomaly"], errors="ignore")
if numeric.shape[1] == 0:
    print("No numeric columns found in dataset.")
    raise SystemExit(1)

X = numeric.values
if "Anomaly" in df.columns:
    y_true = df["Anomaly"].values
else:
    print("No Anomaly column found for evaluation.")
    raise SystemExit(1)

clf = joblib.load(MODEL_FILE)
y_pred = clf.predict(X)
# Convert 1 -> normal (0), -1 -> anomaly (1)
y_pred = np.where(y_pred == 1, 0, 1)

print("Confusion Matrix (One-Class SVM):")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')
plt.title("One-Class SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Normal", "Anomaly"])
plt.yticks([0,1], ["Normal", "Anomaly"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='red')
plt.savefig(OUTPUT_FILE)
plt.close()
print("Confusion matrix saved to", OUTPUT_FILE)
