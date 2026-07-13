#!/usr/bin/env python3
"""Generate every dataset for the primary Gaussian graph-recovery study.

This file is intentionally a no-argument entry point.  Running it directly in
PyCharm creates the seven configurations and ten replications specified in the
manuscript, together with one manifest describing all 70 instances.
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


STUDY = "primary_graph_recovery"
OUTPUT_ROOT = PROJECT_DIR / "data" / "gaussian_experiments"
BASE_SEED = 2027
NUMBER_OF_REPLICATIONS = 10
TARGET_DEGREE = 4
TARGET_SIGNAL = 0.20
TARGET_CONDITION = 10.0
OVERWRITE_EXISTING = False

# Panel A varies n at p=40. Panel B varies p at n=2p. Panel C adds a
# scale-free graph at the shared central setting.  The dictionary construction
# removes the repeated Erd--Renyi configuration (p, n)=(40, 80).
PANEL_SETTINGS = {
    "sample_size": [
        {"topology": "erdos_renyi", "p": 40, "n": n}
        for n in (20, 40, 80, 160)
    ],
    "dimension": [
        {"topology": "erdos_renyi", "p": p, "n": 2 * p}
        for p in (20, 40, 60)
    ],
    "topology": [
        {"topology": "erdos_renyi", "p": 40, "n": 80},
        {"topology": "scale_free", "p": 40, "n": 80},
    ],
}


def unique_settings() -> list[dict[str, int | str]]:
    """Return the seven configurations without repeated panel entries."""
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
        "target_degree": TARGET_DEGREE,
        "target_signal": TARGET_SIGNAL,
        "target_condition": TARGET_CONDITION,
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
    if panel not in PANEL_SETTINGS:
        choices = ", ".join(sorted(PANEL_SETTINGS))
        raise ValueError(f"unknown panel {panel!r}; choose one of {choices}")

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
