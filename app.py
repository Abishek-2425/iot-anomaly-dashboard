# app.py

from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
BASE = Path(__file__).parent

# Files
DATA_FILE = BASE / "iot_network_data.csv"
MODEL_FILE = BASE / "model.pkl"

# Global runtime state
df_source = pd.DataFrame()
model = None
top_devices = []
device_state = {}     # live simulated state (temp/hum/bat)
TOP_K = 5


# ----------------------------------------------------
# Helper: Identify sensor feature names in dataset
# ----------------------------------------------------
def detect_features(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_map = {
        "temp": None,
        "hum": None,
        "bat": None
    }

    for candidates, key in [
        (["Temperature", "Temp", "temperature", "latency_ms"], "temp"),
        (["Humidity", "Hum", "humidity", "jitter_ms"], "hum"),
        (["Battery_Level", "Battery", "battery", "throughput_kbps"], "bat")
    ]:
        for c in candidates:
            if c in numeric:
                feature_map[key] = c
                break

    # fallback: just use numeric columns in order
    if feature_map["temp"] is None and len(numeric) > 0:
        feature_map["temp"] = numeric[0]
    if feature_map["hum"] is None and len(numeric) > 1:
        feature_map["hum"] = numeric[1]
    if feature_map["bat"] is None and len(numeric) > 2:
        feature_map["bat"] = numeric[2]

    return feature_map


# ----------------------------------------------------
# Load dataset & initialize device simulation
# ----------------------------------------------------
def load_dataset():
    global df_source, top_devices, device_state

    if not DATA_FILE.exists():
        df_source = pd.DataFrame()
        top_devices = []
        device_state = {}
        return

    df_source = pd.read_csv(DATA_FILE)

    # Ensure Device_ID exists
    if "Device_ID" not in df_source.columns:
        df_source.insert(0, "Device_ID", [f"Device_{i%50+1}" for i in range(len(df_source))])

    # Pick top devices based on frequency
    counts = df_source["Device_ID"].value_counts()
    top_devices = list(counts.index[:TOP_K])

    features = detect_features(df_source)

    # Base values = mean per device for realism
    device_state = {}
    for dev in top_devices:
        subset = df_source[df_source["Device_ID"] == dev]
        means = subset.mean(numeric_only=True) if not subset.empty else {}

        device_state[dev] = {
            "temp_key": features["temp"],
            "hum_key": features["hum"],
            "bat_key": features["bat"],
            "temp": float(means.get(features["temp"], 25.0)),
            "hum": float(means.get(features["hum"], 45.0)),
            "bat": float(means.get(features["bat"], 80.0))
        }


load_dataset()


# ----------------------------------------------------
# Try loading model
# ----------------------------------------------------
if MODEL_FILE.exists():
    try:
        model = joblib.load(MODEL_FILE)
        print("✅ Loaded model.pkl")
    except Exception:
        print("⚠️ Failed to load model, will rebuild on upload.")
        model = None


# ----------------------------------------------------
# Simulated Live Stream
# ----------------------------------------------------
np.random.seed(7)

def generate_live_data():
    """Simulate new readings for each top device."""
    rows = []
    for dev, st in device_state.items():

        # Natural drift + occasional anomalies
        st["temp"] += np.random.normal(0, 0.15)
        st["hum"] += np.random.normal(0, 0.20)
        st["bat"] -= abs(np.random.normal(0.03, 0.01))

        if np.random.rand() < 0.06:  # inject anomaly
            st["temp"] += np.random.uniform(5, 20)
            st["hum"] += np.random.uniform(10, 35)
            st["bat"] -= np.random.uniform(5, 20)

        st["bat"] = max(st["bat"], 0)

        row = {
            "Device_ID": dev,
            st["temp_key"]: round(st["temp"], 3),
            st["hum_key"]: round(st["hum"], 3),
            st["bat_key"]: round(st["bat"], 3)
        }
        rows.append(row)

    return rows


# ----------------------------------------------------
# Flask Routes
# ----------------------------------------------------
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/upload", methods=["POST"])
def upload():
    global model
    file = request.files.get("file")

    if not file:
        return jsonify({"success": False, "error": "No file provided"}), 400

    # Save and re-load dataset
    save_path = BASE / "uploaded.csv"
    file.save(save_path)

    try:
        df = pd.read_csv(save_path)
    except Exception:
        return jsonify({"success": False, "error": "Invalid CSV file"}), 400

    df.to_csv(DATA_FILE, index=False)
    load_dataset()

    numeric = df.select_dtypes(include=[np.number]).drop(columns=["Anomaly"], errors="ignore")
    if numeric.empty:
        return jsonify({"success": False, "error": "Dataset contains no numeric columns"}), 400

    # Prefer known normal rows
    if "Anomaly" in df.columns and (df["Anomaly"] == 0).sum() >= 10:
        train_data = df[df["Anomaly"] == 0].select_dtypes(include=[np.number]).drop(columns=["Anomaly"], errors="ignore").values
    else:
        train_data = numeric.values

    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    clf.fit(train_data)

    joblib.dump(clf, MODEL_FILE)
    model = clf

    return jsonify({"success": True, "message": "Dataset uploaded and model trained successfully."})


@app.route("/data/latest")
def live_data():
    if df_source.empty or not top_devices:
        return jsonify({"success": False, "error": "No dataset loaded."}), 400

    rows = generate_live_data()
    df_live = pd.DataFrame(rows)

    numeric = df_live.select_dtypes(include=[np.number]).values
    preds = model.predict(numeric) if model is not None else [1] * len(df_live)

    output = []
    for r, p in zip(rows, preds):
        r["anomaly"] = 1 if p == -1 else 0
        output.append(r)

    return jsonify({"success": True, "rows": output})


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE / "static", filename)


# ----------------------------------------------------
# Run
# ----------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting Dashboard at http://localhost:5000")
    app.run(debug=True, port=5000)
