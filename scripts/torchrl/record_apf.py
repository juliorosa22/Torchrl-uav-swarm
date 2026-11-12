"""Record the plain-APF controller (no learning) packing on the PackingMono task."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="PackingMono-TorchRL-UAVSwarm-Direct-v0")
parser.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg_local.yaml")
parser.add_argument("--steps", type=int, default=1800)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--kp", type=float, default=2.0)
parser.add_argument("--repel", type=float, default=2.0)
parser.add_argument("--kd", type=float, default=1.0)
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


def camera_pose(centroid, target):
    mid = (centroid + target) / 2
    diff = (target - centroid)[:2]
    dist = diff.norm().clamp(min=1e-3)
    perp = torch.stack([-diff[1], diff[0]]) / dist
    offset = max(3.5, 0.8 * dist.item())
    eye = mid.clone()
    eye[0] += perp[0] * offset
    eye[1] += perp[1] * offset
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
    cfg.sim.render.enable_translucency = True
    env = gym.make(args_cli.task, cfg=cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, args_cli.video_dir, step_trigger=lambda s: s == 0, video_length=args_cli.steps, disable_logger=True
    )
    u = env.unwrapped
    names = [f"robot_{i}" for i in range(u.num_drones)]
    env.reset()
    u.episode_length_buf[:] = 0

    def pos():
        return torch.stack([r.data.root_pos_w for r in u._robots], dim=1)[0]

    def aim():
        eye, look = camera_pose(pos().mean(0), u._desired_pos_w[0, 0])
        u.sim.set_camera_view(eye, look)

    aim()
    episodes = goals = 0
    with torch.no_grad():
        for step in range(args_cli.steps):
            act = compute_swarm_gravity_baseline_action(u, args_cli.kp, args_cli.repel)
            env.step({n: act[:, i] for i, n in enumerate(names)})
            if step % 15 == 0:
                aim()
            if bool(u._last_terminated[0] | u._last_timed_out[0]):
                episodes += 1
                goals += int(u._termination_reasons["goal_reached"][0])
                print(f"[rec] step {step}: episode {episodes} ended, goal={bool(u._termination_reasons['goal_reached'][0])}", flush=True)
                aim()
    print(f"[rec] DONE episodes={episodes} goals={goals}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
