#!/usr/bin/env python3
"""Generate every dataset for the primary Gaussian graph-recovery study.

This no-argument entry point creates ten registered p=500, n=1000 ROC datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable


# Permit direct execution from PyCharm even when the working directory is not
# the repository root.  Module execution (``python -m experiments...``) does
# not need this branch.
if __package__ in {None, ""}:
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_DIR))
else:
    PROJECT_DIR = Path(__file__).resolve().parents[1]

from experiments.generate_gaussian_experiments import generate_one
from experiments.primary_graph_recovery_config import (
    NUMBER_OF_REPLICATIONS,
    PANEL_SETTINGS as REGISTERED_PANEL_SETTINGS,
)


STUDY = "primary_graph_recovery"
OUTPUT_ROOT = PROJECT_DIR / "data" / "gaussian_experiments"
BASE_SEED = 2027
# The generic generator requires these arguments, but the registered
# lattice-with-hubs branch follows Lin et al. directly and does not calibrate
# its population to them.
TARGET_DEGREE = 4
TARGET_SIGNAL = 0.20
TARGET_CONDITION = 10.0
OVERWRITE_EXISTING = False


def _setting(configuration: tuple[str, int, int]) -> dict[str, int | str]:
    topology, p, n = configuration
    return {"topology": topology, "p": p, "n": n}


PANEL_SETTINGS = {
    panel: [_setting(configuration) for configuration in configurations]
    for panel, configurations in REGISTERED_PANEL_SETTINGS.items()
}


def unique_settings() -> list[dict[str, int | str]]:
    """Return the registered configurations without repeated panel entries."""
    unique: dict[tuple[str, int, int], dict[str, int | str]] = {}
    for settings in PANEL_SETTINGS.values():
        for setting in settings:
            key = (str(setting["topology"]), int(setting["p"]), int(setting["n"]))
            unique[key] = setting
    return list(unique.values())


def generate_all(
    *,
    output_root: Path = OUTPUT_ROOT,
    replications: Iterable[int] = range(NUMBER_OF_REPLICATIONS),
    overwrite: bool = OVERWRITE_EXISTING,
) -> list[dict[str, object]]:
    """Generate the full design and return its instance records."""
    settings = unique_settings()
    replication_ids = list(replications)
    records: list[dict[str, object]] = []

    print(
        f"Generating {len(settings)} settings x {len(replication_ids)} replications "
        f"under {output_root / STUDY}"
    )
    for setting in settings:
        for rep in replication_ids:
            record = generate_one(
                study=STUDY,
                topology=str(setting["topology"]),
                p=int(setting["p"]),
                n=int(setting["n"]),
                target_degree=TARGET_DEGREE,
                target_signal=TARGET_SIGNAL,
                target_condition=TARGET_CONDITION,
                rep=rep,
                base_seed=BASE_SEED,
                output_root=output_root,
                overwrite=overwrite,
                fixed_graph=False,
            )
            records.append(record)
            print(
                f"{record['generation_status']}: {record['topology']} "
                f"p={record['p']} n={record['n']} rep={record['rep']}"
            )

    study_directory = output_root / STUDY
    study_directory.mkdir(parents=True, exist_ok=True)
    design = {
        "study": STUDY,
        "number_of_unique_settings": len(settings),
        "number_of_replications": len(replication_ids),
        "number_of_instances": len(records),
        "base_seed": BASE_SEED,
        "data_generation": "Lin, Drton, and Shojaie (2016), Section 4.1; five components for p=500",
        "paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5476334/",
        "panels": PANEL_SETTINGS,
        "unique_settings": settings,
    }
    (study_directory / "design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (study_directory / "manifest.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {study_directory / 'design.json'}")
    print(f"Wrote {study_directory / 'manifest.json'} with {len(records)} instances")
    return records


def generate_panel(
    panel: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    replications: Iterable[int] = range(NUMBER_OF_REPLICATIONS),
    overwrite: bool = OVERWRITE_EXISTING,
    manifest_name: str | None = None,
) -> list[dict[str, object]]:
    """Generate one manuscript panel and write a panel-specific manifest."""
    replication_ids = list(replications)
    records: list[dict[str, object]] = []
    for setting in PANEL_SETTINGS[panel]:
        for rep in replication_ids:
            record = generate_one(
                study=STUDY,
                topology=str(setting["topology"]),
                p=int(setting["p"]),
                n=int(setting["n"]),
                target_degree=TARGET_DEGREE,
                target_signal=TARGET_SIGNAL,
                target_condition=TARGET_CONDITION,
                rep=rep,
                base_seed=BASE_SEED,
                output_root=output_root,
                overwrite=overwrite,
                fixed_graph=False,
            )
            records.append(record)
            print(
                f"{record['generation_status']}: {record['topology']} "
                f"p={record['p']} n={record['n']} rep={record['rep']}"
            )

    study_directory = output_root / STUDY
    study_directory.mkdir(parents=True, exist_ok=True)
    manifest_stem = manifest_name or f"manifest_{panel}"
    manifest = study_directory / f"{manifest_stem}.json"
    manifest.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest} with {len(records)} instances")
    return records


def main() -> None:
    generate_all()


if __name__ == "__main__":
    main()
