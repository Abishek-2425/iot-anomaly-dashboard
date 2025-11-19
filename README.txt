
IoT Anomaly Dashboard (Flask + Chart.js) - Cyber Dark Theme
Project directory: /mnt/data/iot-anomaly-dashboard

Files included:
- app.py               : Flask backend serving dashboard + /data/latest endpoint
- model_train.py      : training script (already included)
- templates/dashboard.html
- static/style.css
- static/charts.js

How to run locally:
1. Install dependencies:
   pip install flask scikit-learn pandas joblib

2. Place a CSV 'synthetic_iot_dataset.csv' in the same folder (or upload via UI) and optionally run:
   python model_train.py

3. Run the app:
   python app.py

4. Open browser at http://localhost:5000

Notes:
- 'Upload & Train' allows you to upload a new CSV. The server will train IsolationForest on the numeric columns and use the uploaded data as the live source.
- The dashboard simulates live IoT streaming by rotating through the CSV rows every 1.2 seconds.
