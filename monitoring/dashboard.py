import os
import webbrowser

def open_dashboard():
    """Launches the generated Evidently Data Drift Report in the browser."""
    report_path = os.path.abspath("monitoring/data_drift_report.html")
    if os.path.exists(report_path):
        print(f"Opening Data Drift Report: {report_path}")
        webbrowser.open(f"file://{report_path}")
    else:
        print("Data Drift Report HTML not found. Please run drift_detector.py first.")

if __name__ == "__main__":
    open_dashboard()