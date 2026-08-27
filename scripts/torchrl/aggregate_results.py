# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Aggregate multi-seed MAPPO training results into paper-ready statistics.

Reads TensorBoard scalar logs from every seed run produced by
scripts/torchrl/train_multi_seed.py and reports, per metric: mean, std, median, the
interquartile mean (IQM), and a bootstrapped 95% confidence interval across seeds.

Follows Agarwal et al. 2021, "Deep Reinforcement Learning at the Edge of the Statistical
Precipice" (NeurIPS) -- IQM is less outlier-sensitive than the mean and more statistically
efficient than the median; interval estimates matter more than point estimates when the
seed count is small. Reimplemented here with plain numpy rather than depending on the
`rliable` package, since we're aggregating one task across seeds, not many tasks -- the
per-task stratified bootstrap `rliable` is built for doesn't apply here.

No Isaac Sim import -- this only reads TensorBoard event files, so it can run outside the
isaaclab conda env (just needs numpy + tensorboard, both already present as MAPPO deps).

Usage:
  # aggregate every seed dir matching the glob, using each one's most recent run
  python scripts/torchrl/aggregate_results.py \\
      --experiment_glob "logs/torchrl/formation_scalability_seed*"

  # also dump the raw per-seed, per-step curves for plotting mean +/- CI bands
  python scripts/torchrl/aggregate_results.py \\
      --experiment_glob "logs/torchrl/formation_scalability_seed*" \\
      --out_csv results/formation_training_seeds.csv
"""

import argparse
import csv
import glob
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_METRICS = [
    "Episode_Termination/goal_reached",
    "Metrics/final_distance_to_goal",
    "Episode_Reward/swarm_cohesion",
    "Episode_Termination/collision",
    "Reward/Average",
]


def find_run_dirs(experiment_glob: str, all_runs: bool = False) -> list[str]:
    """Resolve <experiment_glob> (matching <experiment_directory>_seed<N> dirs) to a list
    of TensorBoard log directories -- one per seed by default (its most recent run), or
    every run inside every matched seed dir if all_runs=True.
    """
    seed_dirs = sorted(d for d in glob.glob(experiment_glob) if os.path.isdir(d))
    run_dirs = []
    for seed_dir in seed_dirs:
        timestamped = sorted(d for d in glob.glob(os.path.join(seed_dir, "*")) if os.path.isdir(d))
        if not timestamped:
            continue
        chosen = timestamped if all_runs else [timestamped[-1]]
        for run_dir in chosen:
            log_dir = os.path.join(run_dir, "logs")
            if os.path.isdir(log_dir):
                run_dirs.append(log_dir)
    return run_dirs


def load_scalar(log_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def iqm(values: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% of values."""
    values = np.sort(np.asarray(values))
    n = len(values)
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    trimmed = values[lo:hi] if hi > lo else values
    return float(trimmed.mean())


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, ci: float = 95.0, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the IQM across seeds."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        boot_stats[i] = iqm(rng.choice(values, size=n, replace=True))
    lo = float(np.percentile(boot_stats, (100 - ci) / 2))
    hi = float(np.percentile(boot_stats, 100 - (100 - ci) / 2))
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiment_glob", required=True,
        help='Glob matching seed dirs, e.g. "logs/torchrl/formation_scalability_seed*".',
    )
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="TensorBoard scalar tags to aggregate.")
    parser.add_argument("--final_window", type=int, default=10, help="Trailing logged points averaged per seed.")
    parser.add_argument("--all_runs", action="store_true", help="Use every run in each seed dir, not just the latest.")
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--out_csv", type=str, default=None, help="If set, also write summary + raw curves CSVs.")
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.experiment_glob, args.all_runs)
    if not run_dirs:
        raise SystemExit(f"No run directories matched {args.experiment_glob!r}")

    print(f"\n[INFO] Aggregating {len(run_dirs)} seed run(s):")
    for d in run_dirs:
        print(f"   {d}")

    per_metric_finals: dict[str, list[float]] = {m: [] for m in args.metrics}
    curve_rows: list[list] = []

    for run_dir in run_dirs:
        for metric in args.metrics:
            result = load_scalar(run_dir, metric)
            if result is None:
                print(f"[WARN] tag {metric!r} not found in {run_dir}")
                continue
            steps, values = result
            final_val = float(values[-args.final_window:].mean()) if len(values) else float("nan")
            per_metric_finals[metric].append(final_val)
            if args.out_csv:
                curve_rows.extend([run_dir, metric, int(s), float(v)] for s, v in zip(steps, values))

    n_seeds = len(run_dirs)
    print(f"\n{'=' * 100}")
    print(f"  Final performance across {n_seeds} seed(s)  (last {args.final_window} logged points averaged per seed)")
    print(f"{'=' * 100}")
    print(f"  {'Metric':<38}{'n':>3}{'mean':>10}{'std':>10}{'median':>10}{'IQM':>10}{'95% CI (IQM)':>22}")

    summary_rows = []
    for metric, finals in per_metric_finals.items():
        finals_arr = np.array([f for f in finals if not np.isnan(f)])
        if len(finals_arr) == 0:
            print(f"  {metric:<38}  (no data)")
            continue
        mean, std, median = float(finals_arr.mean()), float(finals_arr.std()), float(np.median(finals_arr))
        iqm_val = iqm(finals_arr)
        if len(finals_arr) > 1:
            ci_lo, ci_hi = bootstrap_ci(finals_arr, n_boot=args.n_boot)
        else:
            ci_lo, ci_hi = iqm_val, iqm_val
        print(
            f"  {metric:<38}{len(finals_arr):>3}{mean:>10.4f}{std:>10.4f}{median:>10.4f}{iqm_val:>10.4f}"
            f"   [{ci_lo:.4f}, {ci_hi:.4f}]"
        )
        summary_rows.append([metric, len(finals_arr), mean, std, median, iqm_val, ci_lo, ci_hi])
    print(f"{'=' * 100}\n")

    if n_seeds < 5:
        print(
            f"[WARN] Only {n_seeds} seed(s) aggregated. The literature floor for a statistically "
            "defensible comparison is 5 seeds (Henderson et al., 'Deep RL that Matters'); "
            "confidence intervals above will be wide and should be treated as provisional.\n"
        )

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        curves_path = os.path.splitext(args.out_csv)[0] + "_curves.csv"

        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "n_seeds", "mean", "std", "median", "iqm", "ci95_lo", "ci95_hi"])
            writer.writerows(summary_rows)

        with open(curves_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_dir", "metric", "step", "value"])
            writer.writerows(curve_rows)

        print(f"[INFO] Wrote summary to {args.out_csv}")
        print(f"[INFO] Wrote raw per-seed curves to {curves_path}")


if __name__ == "__main__":
    main()
