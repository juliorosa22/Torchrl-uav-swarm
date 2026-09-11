"""Geometric controller hover test: command all drones to hold a target altitude.

No learned policy — a simple proportional altitude controller outputs velocity
commands that feed into the SE(3) geometric (or pd_velocity) low-level controller:

    vz_cmd = clip(Kp * (z_target - z_current) / max_lin_vel, -1, 1)
    vx = vy = yaw_rate = 0

Drones spawn at the default stage-1 height (~1 m) and climb to the target.
Use this to verify the controller stabilises altitude before training.

Usage:
    python scripts/torchrl/flight_test.py
    python scripts/torchrl/flight_test.py --num_envs 8 --target_altitude 4.0
    python scripts/torchrl/flight_test.py --controller pd_velocity --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hover flight test for the geometric controller.")
parser.add_argument("--task", type=str, default="FullTask-TorchRL-UAVSwarm-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments to visualise.")
parser.add_argument("--target_altitude", type=float, default=4.0, help="Hover altitude target in metres.")
parser.add_argument(
    "--controller", type=str, default="geometric",
    choices=["geometric", "pd_velocity"],
    help="Low-level controller to test.",
)
parser.add_argument("--kp", type=float, default=2.0, help="Altitude proportional gain.")
parser.add_argument("--vx_cmd", type=float, default=0.0, help="Constant vx_b command in [-1,1], added on top of altitude hold.")
parser.add_argument("--yaw_cmd", type=float, default=0.0, help="Constant yaw_rate command in [-1,1], added on top of altitude hold.")
parser.add_argument("--duration", type=int, default=1000, help="Steps to run (0 = infinite).")
parser.add_argument("--log_interval", type=int, default=50, help="Steps between status prints.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports after sim is up ──────────────────────────────────────────────────
import time
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import DirectMARLEnv

import UavSwarm.tasks  # noqa: F401


# ── hover controller ─────────────────────────────────────────────────────────

def hover_actions(env: DirectMARLEnv, target_z: float, max_vel: float, kp: float, vx_cmd: float = 0.0, yaw_cmd: float = 0.0) -> dict:
    """P-altitude controller: proportional vz toward target_z, plus a constant
    vx_b/yaw_rate command to test whether lateral motion destabilizes attitude.

    Returns a dict matching the MARL action format:
        { agent_name: (num_envs, 4) }  — [vx_b, vy_b, vz_b, yaw_rate] in [-1, 1]
    """
    actions = {}
    for i, agent in enumerate(env.cfg.possible_agents):
        z = env._robots[i].data.root_pos_w[:, 2]          # (num_envs,)
        vz_cmd = (kp * (target_z - z) / max_vel).clamp(-1.0, 1.0)
        act = torch.zeros(env.num_envs, 4, device=env.device)
        act[:, 0] = vx_cmd
        act[:, 2] = vz_cmd
        act[:, 3] = yaw_cmd
        actions[agent] = act
    return actions


# ── diagnostics helper ────────────────────────────────────────────────────────

def read_state(env: DirectMARLEnv):
    """Return (mean_z, min_z, max_z, mean_vz, xy_drift, tilt_deg) across all drones/envs.

    xy_drift is relative to each env's own origin (env.scene.env_origins), not world
    origin -- parallel envs are spatially offset, so a raw world-frame xy norm is
    dominated by that spacing rather than actual drift.
    """
    pos_list = [r.data.root_pos_w for r in env._robots]   # each (num_envs, 3)
    vel_list = [r.data.root_lin_vel_b for r in env._robots]
    quat_list = [r.data.root_quat_w for r in env._robots]  # (num_envs, 4) w,x,y,z

    pos = torch.stack(pos_list, dim=0)   # (num_drones, num_envs, 3)
    vel = torch.stack(vel_list, dim=0)   # (num_drones, num_envs, 3)
    quat = torch.stack(quat_list, dim=0)  # (num_drones, num_envs, 4)

    z   = pos[..., 2]
    vz  = vel[..., 2]
    xy_local = pos[..., :2] - env.scene.env_origins[:, :2].unsqueeze(0)

    # Tilt angle from vertical: angle between body-z axis and world-z, derived from quaternion.
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    b3_z = 1.0 - 2.0 * (qx**2 + qy**2)  # world-z component of the body-z axis
    tilt_deg = torch.rad2deg(torch.acos(b3_z.clamp(-1.0, 1.0)))

    return (
        z.mean().item(),
        z.min().item(),
        z.max().item(),
        vz.mean().item(),
        xy_local.norm(dim=-1).mean().item(),
        tilt_deg.mean().item(),
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  UAV Swarm — Hover Flight Test")
    print("=" * 70)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Stage 1 = hovering only; no obstacles, shorter episodes.
    env_cfg.curriculum.active_stage = 1
    env_cfg.controller.type = args_cli.controller

    max_vel = env_cfg.controller.max_lin_vel_cmd

    print(f"  Task          : {args_cli.task}")
    print(f"  Controller    : {env_cfg.controller.type}")
    print(f"  Kv (ctrl gain): {env_cfg.controller.Kv}")
    print(f"  KR            : {env_cfg.controller.KR}")
    print(f"  KOmega        : {env_cfg.controller.KOmega}")
    print(f"  Max lin vel   : {max_vel} m/s")
    print(f"  Target alt    : {args_cli.target_altitude} m")
    print(f"  Hover Kp      : {args_cli.kp}")
    print(f"  vx_cmd        : {args_cli.vx_cmd}")
    print(f"  yaw_cmd       : {args_cli.yaw_cmd}")
    print(f"  Envs          : {env_cfg.scene.num_envs}")
    print(f"  Spawn height  : {env_cfg.curriculum.spawn_height_range}")
    print()

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped: DirectMARLEnv = env.unwrapped

    env.reset()

    header = f"  {'Step':>6}  {'Alt mean':>9}  {'Alt min':>8}  {'Alt max':>8}  {'Vz mean':>8}  {'XY drift':>9}  {'Tilt deg':>9}  {'Err':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    max_steps = args_cli.duration if args_cli.duration > 0 else int(1e9)
    t0 = time.time()

    for step in range(max_steps):
        acts = hover_actions(unwrapped, args_cli.target_altitude, max_vel, args_cli.kp, args_cli.vx_cmd, args_cli.yaw_cmd)
        env.step(acts)

        if (step + 1) % args_cli.log_interval == 0:
            mean_z, min_z, max_z, mean_vz, xy_drift, tilt_deg = read_state(unwrapped)
            err = abs(mean_z - args_cli.target_altitude)
            print(
                f"  {step+1:>6}  "
                f"{mean_z:>9.3f}  "
                f"{min_z:>8.3f}  "
                f"{max_z:>8.3f}  "
                f"{mean_vz:>8.3f}  "
                f"{xy_drift:>9.3f}  "
                f"{tilt_deg:>9.3f}  "
                f"{err:>7.3f}"
            )

    elapsed = time.time() - t0
    fps = max_steps * env_cfg.scene.num_envs / elapsed
    print(f"\n  Finished {max_steps} steps in {elapsed:.1f} s  ({fps:.0f} env-steps/s).")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  [Interrupted by user]")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
