Here’s the fully updated, complete `README.md` content with all fixes, features, and plug-and-play improvements included:

````markdown
# IoT Anomaly Dashboard (Flask + Chart.js)

**Project Directory:** `d:/PROJECTS/iot-anomaly-dashboard/`

A real-time IoT anomaly detection dashboard using Flask for the backend and Chart.js for frontend visualization. The app simulates live IoT device readings and flags anomalies using multiple models: `IsolationForest`, `One-Class SVM`, and `Autoencoder`.

---

## **Included Files**

* `app.py` — Flask backend serving the dashboard and `/data/latest` endpoint for live data. Handles CSV upload and retraining.
* `train_if.py` — Script to train `IsolationForest` on numeric columns of the dataset (prefers rows where `Anomaly == 0`). Saves model to `models/if_model.pkl`.
* `train_autoencoder.py` — Script to train a simple Autoencoder on numeric columns (prefers rows where `Anomaly == 0`). Saves model to `models/autoencoder_model.keras` in a fully plug-and-play format compatible with modern Keras versions.
* `evaluate_all.py` — Evaluates all models (`IsolationForest`, `One-Class SVM`, `Autoencoder`) on a given dataset. Automatically handles feature selection, scaling, and missing columns. Outputs Accuracy, Precision, Recall, F1 Score, and saves confusion matrices to `output/confusion_matrices_all.png`.
* `models/` — Directory containing pre-trained models (`base_if_model.pkl`, `if_model.pkl`, `autoencoder_model.keras`).
* `templates/dashboard.html` — Dashboard UI for live device monitoring and anomaly alerts.
* `static/style.css` — Styles for the dashboard (cards, charts, table, badges).
* `static/charts.js` — JS for chart updates, live data polling, and alert handling.
* `synthetic_iot_dataset.csv` — Sample IoT dataset with `Device_ID`, `Temperature`, `Humidity`, `Battery_Level`, `Anomaly`.

---

## **Setup & Running Locally**

1. **Install dependencies**

```bash
pip install flask scikit-learn pandas joblib matplotlib seaborn tensorflow
````

2. **Prepare dataset**

* Place `synthetic_iot_dataset.csv` in the project root **or** upload a CSV through the dashboard UI.
* Optional: Train new models manually:

```bash
python train_if.py
python train_autoencoder.py
```

3. **Run the Flask app**

```bash
python app.py
```

4. **Open the dashboard**

* Navigate to [http://localhost:5000](http://localhost:5000) in your browser.

---

## **Features**

* **Live IoT Simulation** — Top 5 devices are updated every second with realistic drift and occasional anomalies.
* **Anomaly Detection** — `IsolationForest`, `One-Class SVM`, and `Autoencoder` flag unusual readings; anomalies appear in the table and trigger alerts.
* **Upload & Train** — Upload a new CSV and retrain models without restarting the server.
* **Charts & Visualization** — Line charts for temperature, humidity, and battery; alerts panel for detected anomalies.
* **Evaluation** — `evaluate_all.py` evaluates all models in one run, calculates Accuracy, Precision, Recall, F1 Score, and saves combined confusion matrices.

---

## **Plug-and-Play Improvements**

* **Autoencoder Compatibility** — The Autoencoder now saves in `.keras` format for modern Keras/TensorFlow compatibility, avoiding legacy H5 deserialization errors.
* **Feature Handling** — Scripts automatically detect numeric columns and exclude the `Anomaly` column. IsolationForest and Autoencoder dynamically adapt to dataset features, preventing "feature mismatch" errors.
* **Scaling** — Autoencoder training automatically scales numeric inputs using `StandardScaler`, ensuring consistent predictions during evaluation.
* **Evaluation Robustness** — `evaluate_all.py` handles missing columns, extra columns, and automatically selects the proper feature subset for each model.
* **Model Replacement Safe** — Pre-trained models can be swapped with new ones safely without breaking the evaluation or dashboard pipeline.

---

## **Notes**

* The simulation uses mean values from the top devices to initialize readings and adds small random variations per polling cycle.
* Anomalies are injected probabilistically in simulated readings.
* The dashboard updates every second and keeps a rolling window of the last 40 readings.
* Evaluation metrics for Autoencoder, IsolationForest, and One-Class SVM are saved and visualized automatically.
* Designed for easy extension: add more anomaly detection models or IoT metrics without changing the core pipeline.

---

## **Evaluation Results Example**

After running `evaluate_all.py` on the default dataset:

| Model           | Accuracy | Precision | Recall | F1 Score |
| --------------- | -------- | --------- | ------ | -------- |
| IsolationForest | 0.9545   | 0.6818    | 1.0000 | 0.8108   |
| One-Class SVM   | 0.6120   | 0.2008    | 1.0000 | 0.3345   |
| Autoencoder     | 0.9525   | 1.0000    | 0.5128 | 0.6780   |

Confusion matrices are saved in `output/confusion_matrices_all.png`.

---

## **Folder Structure Overview**

```
iot-anomaly-dashboard/
│
├─ app.py
├─ train_if.py
├─ train_autoencoder.py
├─ evaluate_all.py
├─ synthetic_iot_dataset.csv
├─ requirements.txt
├─ models/
│   ├─ if_model.pkl
│   └─ autoencoder_model.keras
├─ templates/
│   └─ dashboard.html
├─ static/
│   ├─ style.css
│   └─ charts.js
└─ output/
    └─ confusion_matrices_all.png
```