# train_autoencoder.py
"""
Train a simple Autoencoder on synthetic_iot_dataset.csv.
Prefers rows where Anomaly==0 (if column exists).
Saves autoencoder_model.keras to models/ directory.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# ==== CONFIG ====
BASE = Path(__file__).parent
DATA_FILE = BASE.parent / "synthetic_iot_dataset.csv"
MODEL_FILE = BASE.parent / "models/autoencoder_model.keras"

# ==== LOAD DATA ====
if not DATA_FILE.exists():
    print("Dataset not found at", DATA_FILE)
    raise SystemExit(1)

df = pd.read_csv(DATA_FILE)
numeric = df.select_dtypes(include=[int, float]).drop(columns=["Anomaly"], errors="ignore")
if numeric.shape[1] == 0:
    print("No numeric columns found in dataset.")
    raise SystemExit(1)

if "Anomaly" in df.columns:
    normal = df[df["Anomaly"] == 0].select_dtypes(include=[int, float])
    if normal.shape[0] >= 10:
        X = normal.values
        print(f"Training on {normal.shape[0]} normal rows (Anomaly==0).")
    else:
        X = numeric.values
        print("Not enough normal-labeled rows; training on all numeric rows.")
else:
    X = numeric.values
    print("No Anomaly column - training on all numeric rows.")

# ==== SCALE DATA ====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==== BUILD AUTOENCODER ====
input_dim = X_scaled.shape[1]
model = Sequential([
    Dense(16, activation='relu', input_shape=(input_dim,)),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(input_dim, activation='linear')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# ==== TRAIN ====
model.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=1)

# ==== SAVE MODEL ====
model.save(MODEL_FILE)
print("Trained Autoencoder saved to", MODEL_FILE)
