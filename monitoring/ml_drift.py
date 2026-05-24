import pandas as pd
import os
from evidently import Report
from evidently.presets import DataDriftPreset

def generate_drift_report():
    data_path = "data/processed/bank_marketing_validated.csv"
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Run ingest.py first.")
        return

    df = pd.read_csv(data_path)

    # Split dataset: simulate reference and current data
    # Reference is first 80%, current is last 20%
    split_idx = int(len(df) * 0.8)
    reference = df.iloc[:split_idx].copy()
    current = df.iloc[split_idx:].copy()

    # Introduce artificial drift in 'current' to trigger the drift alert
    print("Introducing artificial drift to 'age' and 'euribor3m' features...")
    current['age'] = current['age'] + 15
    current['euribor3m'] = current['euribor3m'] * 1.5
    
    # Drop the target column to focus only on feature drift
    if 'y' in reference.columns:
        reference = reference.drop(columns=['y'])
        current = current.drop(columns=['y'])

    # Initialize and run Evidently Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    snapshot = drift_report.run(reference_data=reference, current_data=current)
    
    out_dir = "monitoring/evidently_report"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "ml_drift_report.html")
    snapshot.save_html(out_file)
    print(f"ML Drift report successfully generated and saved to {out_file}")

if __name__ == "__main__":
    generate_drift_report()
