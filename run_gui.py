#!/usr/bin/env python3
"""
run_gui.py
==========
Entry point for the URF/A-SSR Real-Time BCI Dashboard.

Usage:
  # Replay adapted Gaddy recording
  python3 run_gui.py --mode replay --file data/processed/Gaddy/Subject_01/Session_01/raw_signals.csv

  # Run synthetic mock stream
  python3 run_gui.py --mode mock

  # Custom confidence threshold
  python3 run_gui.py --mode replay --file data/processed/Gaddy/Subject_01/Session_01/raw_signals.csv --threshold 0.75
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Handle headless environments safely
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication
from backend.gui.app import BCIDashboardWindow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch URF/A-SSR Real-Time Assistive BCI & Silent Speech GUI Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["replay", "mock", "live"],
        default="replay",
        help="System operating mode",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional initial CSV recording path to load",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Path to trained model weights directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Safety confidence threshold for intent gating",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run offscreen for automated validation and testing",
    )
    args = parser.parse_args()

    if args.headless:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    file_path = Path(args.file) if args.file else None

    app = QApplication(sys.argv)
    app.setApplicationName("URF/A-SSR BCI Dashboard")
    app.setOrganizationName("DeepMind / URF")

    window = BCIDashboardWindow(
        mode=args.mode,
        file_path=file_path,
        models_dir=Path(args.models_dir),
        threshold=args.threshold,
    )
    window.show()

    # If running headless/offscreen via flag or CI, don't block indefinitely
    if args.headless:
        print("INFO: Running in headless/offscreen mode (UI initialized successfully).")
        app.processEvents()
        sys.exit(0)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
