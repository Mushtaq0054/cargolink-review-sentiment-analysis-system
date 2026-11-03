"""
CargoLink Review Sentiment Analysis System
Streamlit entrypoint proxy redirecting to dashboard.py
"""
import runpy
import os

if __name__ == "__main__":
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    runpy.run_path(dashboard_path, run_name="__main__")

