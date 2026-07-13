#!/usr/bin/env python3
"""Run the sample-size panel; a no-argument run is a small local test."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.primary_panel_workflow import run_panel_cli


if __name__ == "__main__":
    run_panel_cli("sample_size")

