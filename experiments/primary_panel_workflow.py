"""Shared command-line workflow for the primary graph-recovery panels."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.Run_gaussian_experiments import parse_args as parse_runner_args
from experiments.Run_gaussian_experiments import run as run_gaussian_experiments
from experiments.common import parse_list
from experiments.generate_primary_graph_recovery_data import (
    OUTPUT_ROOT,
    STUDY,
    generate_panel,
)
from experiments.primary_graph_recovery_config import NUMBER_OF_REPLICATIONS


PROJECT_DIR = Path(__file__).resolve().parents[1]


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


def run_panel(panel: str, args: argparse.Namespace) -> None:
    """Pass one panel runner's explicit arguments to the fitting engine."""
    runner_argv = [
        "--study",
        STUDY,
        "--job-name",
        args.job_name,
        "--stage",
        args.stage,
        "--method-list",
        args.method_list,
        "--candidate-rule",
        args.candidate_rule,
        "--time-limit",
        str(args.time_limit),
        "--mip-gap",
        str(args.mip_gap),
        "--threads",
        str(args.threads),
        "--big-m-init",
        str(args.big_m_init),
        "--l1-max-iter",
        str(args.l1_max_iter),
        "--l1-tolerance",
        str(args.l1_tolerance),
        "--l1-support-tolerance",
        str(args.l1_support_tolerance),
        "--graphl0-l2",
        str(args.graphl0_l2),
        "--graphl0-m-bound",
        str(args.graphl0_m_bound),
        "--glasso-max-iter",
        str(args.glasso_max_iter),
        "--glasso-tolerance",
        str(args.glasso_tolerance),
        "--data-root",
        str(args.data_root),
        "--rep-list",
        args.rep_list,
    ]
    if args.configuration_list is None:
        runner_argv.extend(
            [
                "--topology-list",
                args.topology,
                "--p-list",
                str(args.p),
                "--n-list",
                str(args.n),
            ]
        )
    else:
        runner_argv.extend(["--configuration-list", args.configuration_list])
    if args.max_instances is not None:
        runner_argv.extend(["--max-instances", str(args.max_instances)])
    if args.screen_size is not None:
        runner_argv.extend(["--screen-size", str(args.screen_size)])
    runner_argv.extend(["--penalty-constant-list", args.penalty_constant_list])
    results_csv = args.results_csv
    if results_csv is None and args.stage == "evaluation":
        # ``*_rep*.csv`` is the default input pattern of the summarizer.  Give
        # both full and subset direct runs a unique, discoverable replication tag.
        replication_tag = "-".join(map(str, parse_list(args.rep_list, int)))
        results_csv = (
            PROJECT_DIR
            / "experiments_results"
            / f"gaussian_{STUDY}_{panel}_{args.job_name}_rep{replication_tag}.csv"
        )
    if results_csv is not None:
        runner_argv.extend(["--results-csv", str(results_csv)])
    if args.overwrite_results:
        runner_argv.append("--overwrite-results")
    if args.verbose:
        runner_argv.append("--verbose")

    run_gaussian_experiments(parse_runner_args(runner_argv))
