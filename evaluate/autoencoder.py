# eval_autoencoder.py
"""
Evaluate trained Autoencoder on test data.
Generates confusion matrix and classification report.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
DATA_FILE = BASE / "synthetic_iot_dataset.csv"
MODEL_FILE = BASE / "models/autoencoder_model.h5"
OUTPUT_FILE = BASE / "outputs/autoencoder.png"

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

autoencoder = load_model(MODEL_FILE)
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 95)  # top 5% as anomalies
y_pred = (mse > threshold).astype(int)

print("Confusion Matrix (Autoencoder):")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')
plt.title("Autoencoder Confusion Matrix")
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
