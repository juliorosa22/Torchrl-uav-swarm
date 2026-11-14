"""Gain-modulation RL on the lidar preview APF: train at --n agents, then evaluate chosen checkpoints on the mixed path distribution."""

import argparse
import glob
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=20)
p.add_argument("--out", type=str, default="results/apf_moving/lab")
p.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg_remote.yaml")
p.add_argument("--gains", type=float, nargs=5, default=[0.7917, 1.9914, 4.4067, 1.0837, 1.2644], help="kp repel kd ff rscale")
p.add_argument("--preview", type=float, nargs=2, default=[0.1622, 0.8586])
p.add_argument("--iters", type=int, default=1392)
p.add_argument("--eval_at", type=int, nargs="+", default=[396, 792, 1392])
p.add_argument("--envs", type=int, default=32)
p.add_argument("--fpb", type=int, default=8192)
p.add_argument("--batch", type=int, default=4096)
p.add_argument("--gain_mod", type=float, default=1.0)
p.add_argument("--init_log_std", type=float, default=-1.0)
p.add_argument("--checkpoint", type=str, default=None)
p.add_argument("--exp", type=str, default="gm")
p.add_argument("--episodes", type=int, default=288)
p.add_argument("--timeout_h", type=float, default=7.0)
a = p.parse_args()

PY = sys.executable
D = ["--move_profiles", "const", "ramp", "stopgo", "--move_path_lens", "20", "30", "--move_z_max", "8", "--move_a_max", "3", "--move_speed_range", "1", "3"]
g = [str(x) for x in a.gains]
pv = [str(x) for x in a.preview]
exp = f"{a.exp}_n{a.n}"


def run(cmd, timeout_h=None):
    print("[gm]", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=False, timeout=None if timeout_h is None else timeout_h * 3600)
    except subprocess.TimeoutExpired:
        print(f"[gm] TIMEOUT {cmd[2]}", flush=True)


train = [PY, "-u", "scripts/torchrl/mappo_train.py", "--task", "PackingMono-TorchRL-UAVSwarm-Direct-v0", "--stage", "13", "--headless",
         "--config", a.config, "--num_agents", str(a.n), "--max_vel", "4.0", "--num_envs", str(a.envs), "--max_iterations", str(a.iters),
         "--frames_per_batch", str(a.fpb), "--batch_size", str(a.batch), "--seed", "0", "--moving", "--obs_sense", "lidar", "--obs_k", "12" if a.n >= 10 else "8",
         "--move_paths", "train", *D, "--policy_arch", "attention", "--residual_rl", "--residual_baseline", "apf", "--gain_mod", str(a.gain_mod),
         "--apf_preview", *pv, "--residual_kp", g[0], "--residual_repel_gain", g[1], "--apf_damping_gain", g[2], "--residual_ff", g[3],
         "--apf_rscale", g[4], "--init_log_std", str(a.init_log_std), "--entropy_coef", "0.001", "--experiment_directory", exp, "--model_name", "gm"]
if a.checkpoint:
    train += ["--checkpoint", a.checkpoint]
run(train, a.timeout_h)

base = [PY, "-u", "scripts/torchrl/apf_tune.py", "--headless", "--config", a.config, "--num_agents", str(a.n), "--max_vel", "4.0", "--num_envs", "64",
        "--move_paths", "all", *D, "--episodes", str(a.episodes), "--final_params", *g, "--preview_params", *pv,
        "--obs_sense", "lidar", "--obs_k", "12" if a.n >= 10 else "8", "--policy_arch", "attention"]
for it in a.eval_at:
    ck = sorted(glob.glob(f"logs/torchrl/{exp}/*/checkpoints/checkpoint_gm_iter_{it}.pt"))
    if ck:
        run([*base, "--gain_mod", str(a.gain_mod), "--policy_ckpt", ck[-1], "--out", f"{a.out}/{exp}_eval{it}.jsonl"])
print("[gm] ALL DONE", flush=True)
