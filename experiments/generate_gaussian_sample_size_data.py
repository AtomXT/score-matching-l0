#!/usr/bin/env python3
"""Generate the sample-size panel of the primary Gaussian experiment."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.primary_panel_workflow import generate_panel_cli


if __name__ == "__main__":
    generate_panel_cli("sample_size")

