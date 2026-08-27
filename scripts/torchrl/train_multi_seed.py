# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Sequential multi-seed MAPPO training orchestrator.

Deep RL results from a single seed aren't statistically defensible (Henderson et al.,
"Deep RL that Matters"; Colas et al., "How Many Random Seeds?") -- changing only the seed
can produce learning curves that don't even look like they came from the same distribution.
This script launches scripts/torchrl/mappo_train.py once per seed so results can be
aggregated across seeds afterward with aggregate_results.py.

Each seed is run as its own subprocess (its own Isaac Sim simulation app instance) rather
than looped inside one long-lived app -- this codebase already relies on one-scene-per-process
elsewhere (see eval_formation_scalability.py's per-swarm-size invocation model), and Isaac
Sim does not reliably support tearing down and rebuilding a scene within one running app.

Each seed's run lands in its own experiment subdirectory
(<experiment_directory>_seed<N>) so aggregate_results.py can glob them back together.

Usage:
  python scripts/torchrl/train_multi_seed.py \\
      --task Formation-TorchRL-UAVSwarm-Direct-v0 \\
      --experiment_directory formation_scalability \\
      --num_seeds 5 --base_seed 0 \\
      --max_iterations 2000 --headless

  # any other scripts/torchrl/mappo_train.py flag (--num_envs, --controller,
  # --stage, --video, ...) is passed through unchanged to every seed's run.

  python scripts/torchrl/aggregate_results.py \\
      --experiment_glob "logs/torchrl/formation_scalability_seed*"
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds to run (literature floor is 5).")
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument(
        "--experiment_directory", type=str, required=True,
        help="Base experiment name; each seed's run lands under "
             "logs/torchrl/<experiment_directory>_seed<N>/.",
    )
    args_cli, passthrough = parser.parse_known_args()

    mappo_train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mappo_train.py")
    results = []

    for i in range(args_cli.num_seeds):
        seed = args_cli.base_seed + i
        seed_experiment_dir = f"{args_cli.experiment_directory}_seed{seed}"
        cmd = [
            sys.executable, mappo_train_path,
            "--seed", str(seed),
            "--experiment_directory", seed_experiment_dir,
        ] + passthrough

        print(f"\n{'=' * 80}")
        print(f"  Seed {i + 1}/{args_cli.num_seeds}  (seed={seed})  ->  logs/torchrl/{seed_experiment_dir}/")
        print(f"{'=' * 80}")
        print(f"  {' '.join(cmd)}\n")

        proc = subprocess.run(cmd)
        results.append((seed, proc.returncode))
        if proc.returncode != 0:
            print(f"[WARN] Seed {seed} exited with code {proc.returncode} -- continuing with remaining seeds.")

    print(f"\n{'=' * 80}")
    print("  Multi-seed run summary")
    print(f"{'=' * 80}")
    for seed, code in results:
        print(f"  seed={seed}: {'OK' if code == 0 else f'FAILED (exit {code})'}")
    print(f"{'=' * 80}\n")

    print("[INFO] Aggregate results with:")
    print(
        f"  python scripts/torchrl/aggregate_results.py "
        f"--experiment_glob \"logs/torchrl/{args_cli.experiment_directory}_seed*\"\n"
    )

    if any(code != 0 for _, code in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
