"""Fixed project paths and simple source-module loading for scripts."""

from __future__ import annotations

import importlib.util
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(current_dir)
SRC_DIR = os.path.join(PROJECT_DIR, "src")


def load_src_module(module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = os.path.join(SRC_DIR, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
