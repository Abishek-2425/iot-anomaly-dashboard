
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd, joblib, numpy as np
from pathlib import Path

app = Flask(__name__)
BASE = Path(__file__).parent

MODEL_PATH = BASE / "model.pkl"
model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

DATA_PATH = BASE / "iot_network_data.csv"
if DATA_PATH.exists():
    df_all = pd.read_csv(DATA_PATH).reset_index(drop=True)
else:
    df_all = pd.DataFrame()

state = {"index": 0}

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    save_path = BASE / "uploaded.csv"
    f.save(save_path)
    df = pd.read_csv(save_path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return jsonify({"success": False, "error": "No numeric columns found in CSV"}), 400
    from sklearn.ensemble import IsolationForest
    X = numeric.values
    clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    clf.fit(X)
    joblib.dump(clf, BASE / "model.pkl")
    global model, df_all
    model = clf
    df_all = df.copy().reset_index(drop=True)
    return jsonify({"success": True, "message": "Uploaded and model trained."})

@app.route('/data/latest')
def latest():
    global state, df_all, model
    if df_all.empty:
        return jsonify({"success": False, "error": "No dataset loaded on server"}), 400
    idx = state["index"]
    row = df_all.iloc[idx].to_dict()
    numeric = df_all.select_dtypes(include=[np.number])
    x = numeric.iloc[idx].values.reshape(1, -1) if numeric.shape[1]>0 else None
    pred = None
    score = None
    if model is not None and x is not None:
        try:
            pred = int(model.predict(x)[0] == -1)
            score = float(model.decision_function(x)[0])
        except Exception:
            pred = None
            score = None
    state["index"] = (idx + 1) % len(df_all)
    return jsonify({"success": True, "index": idx, "row": row, "anomaly": pred, "score": score})

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(str(BASE / 'static'), filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
