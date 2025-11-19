# model_train.py
"""
Train IsolationForest on synthetic_iot_dataset.csv.
Prefers rows where Anomaly==0 (if column exists).
Saves if_model.pkl to project root.
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

BASE = Path(__file__).parent
DATA_FILE = BASE / "synthetic_iot_dataset.csv"
MODEL_FILE = BASE / "models/if_model.pkl"

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

clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
clf.fit(X)
joblib.dump(clf, MODEL_FILE)
print("Trained IsolationForest saved to", MODEL_FILE)
