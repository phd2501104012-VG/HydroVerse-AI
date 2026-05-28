#!/usr/bin/env python
"""HydroVerse AI — Next-Gen Climate Hazard Intelligence Platform

Usage:
    python main.py dashboard    # Launch Streamlit dashboard
    python main.py api          # Launch FastAPI REST API
    python main.py validate     # Run validation pipeline
    python main.py monitor      # Start real-time monitoring
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CFG
from utils import get_logger

logger = get_logger(__name__)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "dashboard":
        _run_dashboard()
    elif command == "api":
        _run_api()
    elif command == "validate":
        _run_validation()
    elif command == "monitor":
        _run_monitor()
    elif command in ("--help", "-h", "help"):
        print(__doc__)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


def _run_dashboard():
    logger.info("Launching HydroVerse AI dashboard...")
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "streamlit", "run",
           os.path.join(os.path.dirname(__file__), "dashboard", "app.py"),
           "--server.port", str(CFG.dashboard.port),
           "--server.headless", "true"]
    subprocess.run(cmd)


def _run_api():
    from api.rest_api import start_api
    logger.info(f"Starting API server on {CFG.api_host}:{CFG.api_port}...")
    start_api(host=CFG.api_host, port=CFG.api_port)


def _run_validation():
    logger.info("Running validation pipeline...")
    from validation.run import run_all_validations

    results = run_all_validations()
    print("Validation results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


def _run_monitor():
    logger.info("Starting real-time monitoring...")
    from realtime.monitor import run_monitor_loop

    run_monitor_loop()


if __name__ == "__main__":
    main()
