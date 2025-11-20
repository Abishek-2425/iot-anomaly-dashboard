# train_if.py
"""
Train IsolationForest on synthetic_iot_dataset.csv at project root.
Saves trained model to models/if_model.pkl.
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

BASE = Path(__file__).parent.parent  # project root
DATA_FILE = BASE.parent / "synthetic_iot_dataset.csv"
MODEL_FILE = BASE.parent / "models/if_model.pkl"

if not DATA_FILE.exists():
    print("Dataset not found at", DATA_FILE)
    raise SystemExit(1)

df = pd.read_csv(DATA_FILE)
numeric = df.select_dtypes(include=[int, float]).drop(columns=["Anomaly"], errors="ignore")
if numeric.shape[1] == 0:
    print("No numeric columns found.")
    raise SystemExit(1)

if "Anomaly" in df.columns:
    normal = df[df["Anomaly"] == 0].select_dtypes(include=[int, float])
    X = normal.values if normal.shape[0] >= 10 else numeric.values
else:
    X = numeric.values

clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
clf.fit(X)
joblib.dump(clf, MODEL_FILE)
print("Trained IsolationForest saved to", MODEL_FILE)
