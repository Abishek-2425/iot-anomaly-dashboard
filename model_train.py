
# Run this script to train a model on iot_network_data.csv and save model.pkl
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

data_path = Path(__file__).parent / "iot_network_data.csv"
df = pd.read_csv(data_path)
numeric = df.select_dtypes(include=[float,int])
X = numeric.values
clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
clf.fit(X)
joblib.dump(clf, Path(__file__).parent / "model.pkl")
print("Trained model saved to model.pkl")
