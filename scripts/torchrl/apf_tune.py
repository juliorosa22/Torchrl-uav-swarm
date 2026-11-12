"""Black-box tuning of the radius-mode APF controller (kp, repel_gain, damping kd) on the packing task.

One Isaac Sim process evaluates many parameter sets by stepping the env with the APF action only.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="PackingMono-TorchRL-UAVSwarm-Direct-v0")
parser.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg_local.yaml")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=128, help="Completed episodes per candidate.")
parser.add_argument("--warmup_steps", type=int, default=1100, help="Discarded steps after reset (startup staggering).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--iterations", type=int, default=8)
parser.add_argument("--popsize", type=int, default=8)
parser.add_argument("--init", type=float, nargs=3, default=[2.0, 2.0, 1.0], metavar=("KP", "REPEL", "KD"))
parser.add_argument("--init_sigma", type=float, default=0.4, help="Initial log-space std.")
parser.add_argument("--out", type=str, default="apf_tune.jsonl")
parser.add_argument("--grid", action="store_true", help="Evaluate --init only, once, and exit.")
parser.add_argument("--stage", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

import sys  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import UavSwarm.tasks  # noqa: F401, E402
from UavSwarm.tasks.direct.torchrl_swarm.controller import compute_swarm_gravity_baseline_action  # noqa: E402

LOG_LO = torch.log(torch.tensor([0.3, 0.2, 0.05]))
LOG_HI = torch.log(torch.tensor([8.0, 16.0, 8.0]))


class Evaluator:
    def __init__(self):
        with open(args_cli.config) as f:
            config = yaml.safe_load(f)
        cfg = parse_env_cfg(args_cli.task, device=config["env"]["device"], num_envs=args_cli.num_envs)
        for k, v in config.get("controller", {}).items():
            setattr(cfg.controller, k, v)
        cfg.controller.type = "geometric"
        cfg.seed = args_cli.seed
        if args_cli.stage is not None:
            cfg.curriculum.active_stage = args_cli.stage
        self.env = gym.make(args_cli.task, cfg=cfg)
        self.u = self.env.unwrapped
        self.names = [f"robot_{i}" for i in range(self.u.num_drones)]
        print(f"[tune] radius mode={self.u.cfg.swarm_cfg.neighborhood_mode} stage={self.u.cfg.curriculum.active_stage}")

    def _step(self, kp, repel):
        act = compute_swarm_gravity_baseline_action(self.u, kp, repel)
        return self.env.step({n: act[:, i] for i, n in enumerate(self.names)})

    @torch.no_grad()
    def evaluate(self, kp, repel, kd):
        self.u.cfg.swarm_cfg.apf_damping_gain = kd
        self.env.reset()
        keys = ["goal_reached", "inter_agent_collision", "collision", "out_of_bounds"]
        counts = {k: 0 for k in keys}
        n_done = n_timeout = 0
        step = 0
        t0 = time.time()
        while n_done < args_cli.episodes:
            self._step(kp, repel)
            step += 1
            if step <= args_cli.warmup_steps:
                continue
            done = self.u._last_terminated | self.u._last_timed_out
            n = int(done.sum())
            if n == 0:
                continue
            n_done += n
            n_timeout += int(self.u._last_timed_out[done].sum())
            reasons = self.u._termination_reasons
            for k in keys:
                if k in reasons:
                    counts[k] += int(reasons[k][done].sum())
            if step > args_cli.warmup_steps + 6000:
                break
        res = {k: counts[k] / max(n_done, 1) for k in keys}
        res.update(timeout=n_timeout / max(n_done, 1), n=n_done, steps=step, secs=round(time.time() - t0, 1))
        return res


def fitness(res):
    return res["goal_reached"] - 2.0 * res["inter_agent_collision"] - 2.0 * res["collision"]


def main():
    ev = Evaluator()
    torch.manual_seed(args_cli.seed)
    mean = torch.log(torch.tensor(args_cli.init))
    sigma = torch.full((3,), args_cli.init_sigma)

    def run(p, tag):
        kp, repel, kd = [float(x) for x in torch.exp(p)]
        res = ev.evaluate(kp, repel, kd)
        row = dict(tag=tag, kp=kp, repel=repel, kd=kd, fit=fitness(res), **res)
        print("[tune]", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        with open(args_cli.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row["fit"]

    if args_cli.grid:
        run(mean, "grid")
        return

    run(mean, "init")
    best_fit, best = -1e9, mean.clone()
    for it in range(args_cli.iterations):
        pop = (mean + sigma * torch.randn(args_cli.popsize, 3)).clamp(LOG_LO, LOG_HI)
        fits = torch.tensor([run(p, f"it{it}") for p in pop])
        order = fits.argsort(descending=True)
        elite = pop[order[: max(args_cli.popsize // 3, 2)]]
        mean = elite.mean(0)
        sigma = (0.5 * sigma + 0.5 * elite.std(0)).clamp(min=0.05)
        if fits.max() > best_fit:
            best_fit, best = float(fits.max()), pop[order[0]].clone()
        print(f"[tune] iter {it} best_fit={best_fit:.3f} best={[round(float(x), 3) for x in torch.exp(best)]} "
              f"mean={[round(float(x), 3) for x in torch.exp(mean)]}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
