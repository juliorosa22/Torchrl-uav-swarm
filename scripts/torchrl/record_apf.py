"""Record the APF controller (no learning) on the packing task, static or moving-sphere."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="PackingMono-TorchRL-UAVSwarm-Direct-v0")
parser.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg_local.yaml")
parser.add_argument("--steps", type=int, default=None, help="Default: one full moving-sphere episode, or 1800 if static.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--kp", type=float, default=2.0)
parser.add_argument("--repel", type=float, default=2.0)
parser.add_argument("--kd", type=float, default=1.0)
parser.add_argument("--ff", type=float, default=0.0)
parser.add_argument("--rscale", type=float, default=1.0)
parser.add_argument("--move_speed", type=float, default=0.0)
parser.add_argument("--move_distance", type=float, default=6.0)
parser.add_argument("--curve_amp", type=float, default=0.0)
parser.add_argument("--planar", action="store_true")
parser.add_argument("--sep", type=float, default=0.5)
parser.add_argument("--video_dir", type=str, default="videos/apf")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

import sys  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import UavSwarm.tasks  # noqa: F401, E402
from UavSwarm.tasks.direct.torchrl_swarm.controller import compute_swarm_gravity_baseline_action  # noqa: E402


def camera_pose(u, centroid, target):
    if args_cli.move_speed > 0:
        d = u._target_vel_w[0, :2] if float(u._target_vel_w[0, :2].norm()) > 0.05 else u._target_dir_w[0, :2]
        d = d / d.norm().clamp(min=1e-6)
        perp = torch.stack([-d[1], d[0]])
        eye = centroid.clone()
        eye[:2] += perp * 5.0
        eye[2] += 2.5
        return eye.tolist(), centroid.tolist()
    mid = (centroid + target) / 2
    diff = (target - centroid)[:2]
    dist = diff.norm().clamp(min=1e-3)
    perp = torch.stack([-diff[1], diff[0]]) / dist
    eye = mid.clone()
    eye[:2] += perp * max(3.5, 0.8 * dist.item())
    eye[2] += max(2.0, 0.4 * dist.item())
    return eye.tolist(), mid.tolist()


def main():
    with open(args_cli.config) as f:
        config = yaml.safe_load(f)
    cfg = parse_env_cfg(args_cli.task, device=config["env"]["device"], num_envs=1)
    for k, v in config.get("controller", {}).items():
        setattr(cfg.controller, k, v)
    cfg.controller.type = "geometric"
    cfg.seed = args_cli.seed
    cfg.swarm_cfg.apf_damping_gain = args_cli.kd
    cfg.curriculum.move_target_speed = args_cli.move_speed
    cfg.curriculum.move_target_distance = args_cli.move_distance
    cfg.curriculum.move_planar = args_cli.planar
    cfg.curriculum.move_curve_amp = args_cli.curve_amp
    cfg.sim.render.enable_translucency = True
    steps = args_cli.steps
    if steps is None:
        steps = 1800 if args_cli.move_speed <= 0 else int((cfg.curriculum.moving_path_length() / args_cli.move_speed + cfg.curriculum.move_settle_s) / 0.02) + 40
    env = gym.make(args_cli.task, cfg=cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, args_cli.video_dir, step_trigger=lambda s: s == 0, video_length=steps, disable_logger=True)
    u = env.unwrapped
    u._apf_radius_scale = args_cli.rscale
    names = [f"robot_{i}" for i in range(u.num_drones)]
    env.reset()
    u.episode_length_buf[:] = 0
    radius = u.cfg.curriculum.get_containment_radius(u.num_drones, u.cfg.swarm_cfg.min_safe_distance)

    def pos():
        return torch.stack([r.data.root_pos_w for r in u._robots], dim=1)[0]

    def aim():
        eye, look = camera_pose(u, pos().mean(0), u._shared_target_w[0])
        u.sim.set_camera_view(eye, look)

    aim()
    n = inside = packed = 0
    min_nn = 99.0
    with torch.no_grad():
        for step in range(steps):
            act = compute_swarm_gravity_baseline_action(u, args_cli.kp, args_cli.repel, ff_gain=args_cli.ff)
            env.step({nm: act[:, i] for i, nm in enumerate(names)})
            if step % 10 == 0:
                aim()
            p = pos()
            d = (u._shared_target_w[0] - p).norm(dim=1)
            pair = (p.unsqueeze(0) - p.unsqueeze(1)).norm(dim=-1) + 10 * torch.eye(u.num_drones, device=p.device)
            nn = float(pair.min())
            n += 1
            ok_in = bool((d <= radius).all())
            inside += ok_in
            packed += ok_in and nn >= args_cli.sep
            min_nn = min(min_nn, nn)
            if bool(u._last_terminated[0] | u._last_timed_out[0]):
                print(f"[rec] episode ended at step {step}", flush=True)
                break
    print(f"[rec] DONE steps={n} inside={inside / n:.2f} packed={packed / n:.2f} min_nn={min_nn:.2f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
