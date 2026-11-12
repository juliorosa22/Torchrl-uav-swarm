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
parser.add_argument("--num_agents", type=int, default=None)
parser.add_argument("--diag", action="store_true", help="Print the pre-timeout swarm geometry for several gain sets.")
parser.add_argument("--packing_density", type=float, default=None)
parser.add_argument("--sweep", action="store_true", help="Stress-test the --init gains over sensing/time-limit conditions.")
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
from UavSwarm.tasks.direct.torchrl_swarm.controller import (  # noqa: E402
    apf_sense_update,
    compute_swarm_gravity_baseline_action,
)

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
        n = args_cli.num_agents
        if n is not None and n != cfg.num_agents:
            cfg.num_agents = n
            cfg.possible_agents = [f"robot_{i}" for i in range(n)]
            cfg.action_spaces = {f"robot_{i}": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for i in range(n)}
            cfg.observation_spaces = {
                f"robot_{i}": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(cfg.single_observation_space,))
                for i in range(n)
            }
            cfg.state_space = n * cfg.single_observation_space
        if args_cli.packing_density is not None:
            cfg.curriculum.stage8_packing_density = args_cli.packing_density
        self.env = gym.make(args_cli.task, cfg=cfg)
        self.u = self.env.unwrapped
        self.names = [f"robot_{i}" for i in range(self.u.num_drones)]
        self.default_len = self.u.max_episode_length
        self.step_dt = self.u.cfg.sim.dt * self.u.cfg.decimation
        print(f"[tune] radius mode={self.u.cfg.swarm_cfg.neighborhood_mode} stage={self.u.cfg.curriculum.active_stage}")

    def _step(self, kp, repel):
        if self.u._apf_sense is not None:
            apf_sense_update(self.u)
        act = compute_swarm_gravity_baseline_action(self.u, kp, repel)
        return self.env.step({n: act[:, i] for i, n in enumerate(self.names)})

    @torch.no_grad()
    def evaluate(self, kp, repel, kd, pos_noise=0.0, vel_noise=0.0, dropout=0.0, delay=0, max_steps=None):
        self.u.cfg.swarm_cfg.apf_damping_gain = kd
        perturbed = pos_noise > 0 or vel_noise > 0 or dropout > 0 or delay > 0
        self.u._apf_sense = (
            dict(pos_noise=pos_noise, vel_noise=vel_noise, dropout=dropout, delay=delay, buf=[]) if perturbed else None
        )
        self.u.cfg.episode_length_s = (max_steps or self.default_len) * self.step_dt
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


@torch.no_grad()
def diagnose(ev, kp, repel, kd, steps=900):
    u = ev.u
    u.cfg.swarm_cfg.apf_damping_gain = kd
    u._apf_sense = None
    u.cfg.episode_length_s = 10_000
    ev.env.reset()
    alive = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)
    for _ in range(steps):
        ev._step(kp, repel)
        alive &= ~(u._last_terminated | u._last_timed_out)
    pos = torch.stack([r.data.root_pos_w for r in u._robots], dim=0)
    vel = torch.stack([r.data.root_lin_vel_w for r in u._robots], dim=0)
    radius = u.cfg.curriculum.get_containment_radius(u.num_drones, u.cfg.swarm_cfg.min_safe_distance)
    dist = (u._shared_target_w.unsqueeze(0) - pos).norm(dim=2)
    pair = (pos.unsqueeze(1) - pos.unsqueeze(0)).norm(dim=-1) + 10 * torch.eye(u.num_drones, device=u.device).unsqueeze(-1)
    nn = pair.min(dim=1).values
    sep = u.cfg.reward_cfg.pack_min_sep_frac * u.cfg.swarm_cfg.min_safe_distance
    inside, spaced, slow = dist <= radius, nn >= sep, vel.norm(dim=2) <= u.cfg.reward_cfg.pack_speed_threshold
    a = alive
    return dict(
        kp=kp, repel=repel, kd=kd, N=u.num_drones, radius=round(radius, 3), alive=int(a.sum()),
        max_dist=round(float(dist.max(0).values[a].mean()), 3), mean_dist=round(float(dist.mean(0)[a].mean()), 3),
        min_nn=round(float(nn.min(0).values[a].mean()), 3), mean_nn=round(float(nn.mean(0)[a].mean()), 3),
        max_speed=round(float(vel.norm(dim=2).max(0).values[a].mean()), 3),
        all_inside=round(float(inside.all(0)[a].float().mean()), 3), all_spaced=round(float(spaced.all(0)[a].float().mean()), 3),
        all_slow=round(float(slow.all(0)[a].float().mean()), 3),
        success=round(float((inside & spaced & slow).all(0)[a].float().mean()), 3),
    )


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

    if args_cli.diag:
        for kp, repel, kd in [(2, 2, 1), (2, 1, 1), (2, 4, 1), (4, 2, 1), (1, 2, 1), (2, 2, 3)]:
            print("[diag]", json.dumps(diagnose(ev, kp, repel, kd)), flush=True)
        return

    if args_cli.sweep:
        kp, repel, kd = [float(x) for x in torch.exp(mean)]
        conds = [("clean", {})]
        conds += [(f"pos_noise={v}", dict(pos_noise=v)) for v in (0.05, 0.1, 0.2, 0.3)]
        conds += [(f"vel_noise={v}", dict(vel_noise=v)) for v in (0.1, 0.3, 0.6)]
        conds += [(f"dropout={v}", dict(dropout=v)) for v in (0.3, 0.6, 0.9)]
        conds += [(f"delay={v}", dict(delay=v)) for v in (2, 5, 10)]
        conds += [(f"max_steps={v}", dict(max_steps=v)) for v in (700, 500, 350)]
        conds += [("combo", dict(pos_noise=0.1, vel_noise=0.3, dropout=0.3, delay=2))]
        for name, cond in conds:
            res = ev.evaluate(kp, repel, kd, **cond)
            row = dict(cond=name, n_agents=ev.u.num_drones, kp=kp, repel=repel, kd=kd, **res)
            print("[sweep]", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            with open(args_cli.out, "a") as f:
                f.write(json.dumps(row) + "\n")
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
    import os

    os._exit(0)
