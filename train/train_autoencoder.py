# train_autoencoder.py
"""
Train an Autoencoder on synthetic_iot_dataset.csv.
Prefers rows where Anomaly==0 (if column exists).
Saves autoencoder_model.h5 to models/ folder.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
DATA_FILE = BASE / "synthetic_iot_dataset.csv"
MODEL_FILE = BASE / "models/autoencoder_model.h5"

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
    X = normal.values if normal.shape[0] >= 10 else numeric.values
else:
    X = numeric.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simple Autoencoder
input_dim = X_scaled.shape[1]
autoencoder = Sequential([
    Dense(16, activation='relu', input_shape=(input_dim,)),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(input_dim, activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
autoencoder.save(MODEL_FILE)
print("Trained Autoencoder saved to", MODEL_FILE)
