# IoT Anomaly Dashboard (Flask + Chart.js)

**Project Directory:** `d:/PROJECTS/iot-anomaly-dashboard/`

A real-time IoT anomaly detection dashboard using Flask for the backend and Chart.js for frontend visualization. The app simulates live IoT device readings and flags anomalies using an `IsolationForest` model.

---

## **Included Files**

* `app.py` — Flask backend serving the dashboard and `/data/latest` endpoint for live data. Handles CSV upload and retraining.
* `train_if.py` — Script to train `IsolationForest` on numeric columns of the dataset (prefers rows where `Anomaly == 0`). Saves model to `models/if_model.pkl`.
* `evaluation.py` — Evaluate a trained model on a dataset, generating metrics and a confusion matrix.
* `models/` — Directory containing pre-trained models (`base_if_model.pkl`, `if_model.pkl`).
* `templates/dashboard.html` — Dashboard UI for live device monitoring and anomaly alerts.
* `static/style.css` — Styles for the dashboard (cards, charts, table, badges).
* `static/charts.js` — JS for chart updates, live data polling, and alert handling.
* `synthetic_iot_dataset.csv` — Sample IoT dataset with `Device_ID`, `Temperature`, `Humidity`, `Battery_Level`, `Anomaly`.

---

## **Setup & Running Locally**

1. **Install dependencies**

```bash
pip install flask scikit-learn pandas joblib matplotlib seaborn
```

2. **Prepare dataset**

* Place `synthetic_iot_dataset.csv` in the project root **or** upload a CSV through the dashboard UI.
* Optional: Train a new model manually:

```bash
python train_if.py
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
* **Anomaly Detection** — `IsolationForest` flags unusual readings; anomalies appear in the table and trigger alerts.
* **Upload & Train** — Upload a new CSV and retrain the model without restarting the server.
* **Charts & Visualization** — Line charts for temperature, humidity, and battery; alerts panel for detected anomalies.
* **Evaluation** — Optional evaluation script calculates Accuracy, Precision, Recall, F1 Score, and confusion matrix.

---

## **Notes**

* The simulation uses mean values from the top devices to initialize readings and adds small random variations per polling cycle.
* Anomalies are injected probabilistically in simulated readings.
* The dashboard updates every second and keeps a rolling window of the last 40 readings.

---
