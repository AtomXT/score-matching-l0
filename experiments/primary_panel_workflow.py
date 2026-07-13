"""Shared command-line workflow for the primary graph-recovery panels."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.Run_gaussian_experiments import parse_args as parse_runner_args
from experiments.Run_gaussian_experiments import run as run_gaussian_experiments
from experiments.common import parse_list
from experiments.generate_gaussian_experiments import generate_one
from experiments.generate_primary_graph_recovery_data import (
    BASE_SEED,
    NUMBER_OF_REPLICATIONS,
    OUTPUT_ROOT,
    PANEL_SETTINGS,
    STUDY,
    TARGET_CONDITION,
    TARGET_DEGREE,
    TARGET_SIGNAL,
    generate_panel,
)


FULL_METHODS = "sm_l0,sm_l1,graphl0,glasso"
LOCAL_TEST_METHODS = "sm_l1"
FULL_PENALTY_MULTIPLIERS = (
    "0.03125,0.044,0.0625,0.088,0.125,0.177,0.25,0.354,"
    "0.5,0.707,1,1.414,2,2.828,4"
)


def generate_panel_cli(panel: str, argv: list[str] | None = None) -> None:
    """Generate all requested replications for one primary-study panel."""
    parser = argparse.ArgumentParser(
        description=f"Generate the {panel.replace('_', ' ')} graph-recovery panel."
    )
    parser.add_argument(
        "--rep-list",
        default=",".join(map(str, range(NUMBER_OF_REPLICATIONS))),
        help="Replication indices; the no-argument default generates all ten.",
    )
    parser.add_argument("--base-output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--manifest-name", default=f"manifest_{panel}")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    generate_panel(
        panel,
        output_root=args.base_output_root,
        replications=parse_list(args.rep_list, int),
        overwrite=args.overwrite,
        manifest_name=args.manifest_name,
    )


def _panel_filter(panel: str) -> str:
    settings = PANEL_SETTINGS[panel]
    return ";".join(
        f"{item['topology']}:{item['p']}:{item['n']}" for item in settings
    )


def _ensure_small_test_data(panel: str, output_root: Path) -> str:
    """Create one tiny deterministic instance used only by a no-argument run."""
    test_study = f"{panel}_test_run"
    topology = "scale_free" if panel == "topology" else "erdos_renyi"
    generate_one(
        study=test_study,
        topology=topology,
        p=8,
        n=16,
        target_degree=2,
        target_signal=0.20,
        target_condition=5.0,
        rep=0,
        base_seed=BASE_SEED,
        output_root=output_root,
        overwrite=False,
        fixed_graph=False,
    )
    return test_study


def run_panel_cli(panel: str, argv: list[str] | None = None) -> None:
    """Run one panel; no arguments execute one quick SM--L1 smoke test."""
    parser = argparse.ArgumentParser(
        description=(
            f"Fit the {panel.replace('_', ' ')} graph-recovery panel. "
            "With no arguments, run one quick local test."
        )
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Result job name. Omit it to run the small automatic test instead.",
    )
    parser.add_argument(
        "--rep-list",
        default=None,
        help="Replication indices for a named full-panel run.",
    )
    parser.add_argument("--method-list", default=FULL_METHODS)
    parser.add_argument("--penalty-multiplier-list", default=FULL_PENALTY_MULTIPLIERS)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--big-m-scale", type=float, default=1.25)
    parser.add_argument("--graphl0-m-bound", type=float, default=100.0)
    parser.add_argument("--data-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    no_argument_test = args.job_name is None
    if no_argument_test:
        study = _ensure_small_test_data(panel, args.data_root)
        job_name = f"{panel}_test_run"
        # Keep the green-button test independent of Gurobi and finish in a few
        # seconds.  Full Quest jobs explicitly request all four competitors.
        method_list = LOCAL_TEST_METHODS
        multiplier_list = "1"
        configuration_list = None
        rep_list = "0"
        max_instances = "1"
        time_limit = min(args.time_limit, 30.0)
        threads = 1
    else:
        study = STUDY
        job_name = args.job_name
        method_list = args.method_list
        multiplier_list = args.penalty_multiplier_list
        configuration_list = _panel_filter(panel)
        rep_list = args.rep_list
        max_instances = None
        time_limit = args.time_limit
        threads = args.threads

    runner_argv = [
        "--study",
        study,
        "--job-name",
        job_name,
        "--method-list",
        method_list,
        "--penalty-multiplier-list",
        multiplier_list,
        "--time-limit",
        str(time_limit),
        "--mip-gap",
        str(args.mip_gap),
        "--threads",
        str(threads),
        "--big-m-scale",
        str(args.big_m_scale),
        "--graphl0-m-bound",
        str(args.graphl0_m_bound),
        "--data-root",
        str(args.data_root),
    ]
    if rep_list is not None:
        runner_argv.extend(["--rep-list", rep_list])
    if configuration_list is not None:
        runner_argv.extend(["--configuration-list", configuration_list])
    if max_instances is not None:
        runner_argv.extend(["--max-instances", max_instances])
    if args.results_csv is not None:
        runner_argv.extend(["--results-csv", str(args.results_csv)])
    if args.overwrite_results or no_argument_test:
        runner_argv.append("--overwrite-results")
    if args.verbose:
        runner_argv.append("--verbose")

    run_gaussian_experiments(parse_runner_args(runner_argv))
