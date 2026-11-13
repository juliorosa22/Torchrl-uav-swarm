"""Sequential per-cell pipeline for the moving-sphere task (one Isaac Sim per GPU, so stages run as subprocesses):
tune APF gains -> evaluate on curved and helix paths -> optionally train the RL residual -> evaluate it against the zero-policy APF."""

import argparse
import glob
import json
import os
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--cells", type=str, default="oracle5,lidar5,oracle20,lidar20", help="comma list of <oracle|lidar><N>")
p.add_argument("--rl_cells", type=str, default="lidar20")
p.add_argument("--out", type=str, default="results/apf_moving/lab")
p.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg_remote.yaml")
p.add_argument("--max_vel", type=float, default=4.0)
p.add_argument("--speed_range", type=str, nargs=2, default=["1.0", "3.0"])
p.add_argument("--path_lens", type=str, nargs="+", default=["20", "30"])
p.add_argument("--eval_episodes", type=int, default=288)
p.add_argument("--iterations", type=int, default=6)
p.add_argument("--rl_iters", type=int, default=300)
p.add_argument("--rl_envs", type=int, default=32)
p.add_argument("--rl_fpb", type=int, default=16384)
p.add_argument("--rl_batch", type=int, default=2048)
p.add_argument("--residual_scale", type=float, default=0.2)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--dry", action="store_true")
a = p.parse_args()

PY = sys.executable
def dist(paths):
    return ["--move_paths", paths, "--move_profiles", "const", "ramp", "stopgo", "--move_path_lens", *a.path_lens,
            "--move_z_max", "8", "--move_a_max", "3", "--move_speed_range", *a.speed_range]


os.makedirs(a.out, exist_ok=True)


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("[chain]", " ".join(cmd), flush=True)
    if not a.dry:
        subprocess.run(cmd, check=False)


def best(path):
    rows = [json.loads(line) for line in open(path)]
    b = max((r for r in rows if "fit" in r), key=lambda r: r["fit"])
    return [b["kp"], b["repel"], b["kd"], b["ff"], b["rscale"]]


for cell in a.cells.split(","):
    sense, n = ("lidar", int(cell[5:])) if cell.startswith("lidar") else ("oracle", int(cell[6:]))
    lid = ["--lidar"] if sense == "lidar" else []
    base = ["--headless", "--config", a.config, "--num_agents", n, "--max_vel", a.max_vel, "--num_envs", 64]
    init = ["0.98", "1.9", "3.4"] if sense == "lidar" else ["2.4", "1.3", "1.5"]
    tune = f"{a.out}/{cell}_tune.jsonl"
    if os.path.exists(tune):
        os.remove(tune)
    run([PY, "-u", "scripts/torchrl/apf_tune.py", *base, *lid, *dist("train"), "--tune_move", "--episodes", 96,
         "--iterations", a.iterations, "--popsize", 8, "--init", *init, "--seed", a.seed, "--out", tune])
    g = best(tune) if not a.dry else [1, 1, 1, 1, 1]
    print(f"[chain] {cell} best gains {g}", flush=True)
    out = f"{a.out}/{cell}_apf_final.jsonl"
    if os.path.exists(out):
        os.remove(out)
    run([PY, "-u", "scripts/torchrl/apf_tune.py", *base, *lid, *dist("all"), "--episodes", a.eval_episodes,
         "--final_params", *g, "--seed", a.seed, "--out", out])
    if cell not in a.rl_cells.split(","):
        continue
    exp = f"rl_{cell}"
    k = 12 if n >= 10 else 8
    run([PY, "-u", "scripts/torchrl/mappo_train.py", "--task", "PackingMono-TorchRL-UAVSwarm-Direct-v0", "--stage", 13, "--headless",
         "--config", a.config, "--num_agents", n, "--max_vel", a.max_vel, "--num_envs", a.rl_envs, "--max_iterations", a.rl_iters,
         "--frames_per_batch", a.rl_fpb, "--batch_size", a.rl_batch, "--seed", a.seed, "--moving", "--obs_sense", sense,
         "--obs_k", k, *dist("train"), "--policy_arch", "attention", "--residual_rl",
         "--residual_baseline", "apf", "--residual_kp", g[0], "--residual_repel_gain", g[1], "--apf_damping_gain", g[2],
         "--residual_ff", g[3], "--apf_rscale", g[4], "--residual_scale", a.residual_scale, "--init_log_std", -1.5,
         "--entropy_coef", 0.001, "--experiment_directory", exp, "--model_name", cell])
    ckpts = sorted(glob.glob(f"logs/torchrl/{exp}/*/checkpoints/checkpoint_{cell}_iter_{a.rl_iters}.pt"))
    ev = [*base, *dist("all"), "--episodes", a.eval_episodes, "--final_params", *g, "--obs_sense", sense, "--obs_k", k,
          "--residual_scale", a.residual_scale, "--seed", a.seed]
    if ckpts or a.dry:
        run([PY, "-u", "scripts/torchrl/apf_tune.py", *ev, "--policy_ckpt", ckpts[-1] if ckpts else "CKPT",
             "--out", f"{a.out}/{cell}_rl_eval.jsonl"])
    run([PY, "-u", "scripts/torchrl/apf_tune.py", *ev, "--zero_policy", "--out", f"{a.out}/{cell}_zero_eval.jsonl"])
print("[chain] ALL DONE", flush=True)
