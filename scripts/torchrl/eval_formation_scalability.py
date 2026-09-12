# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Zero-shot swarm-size scalability evaluation for the formation-assignment task.

Loads a MAPPO checkpoint trained at one swarm size (e.g. num_agents=5) and evaluates it,
WITHOUT any retraining, on an env instantiated with a different swarm size. This works
because the policy network only depends on obs_dim/action_dim (fixed at 28/4 regardless
of swarm size, see mappo_train.py::make_policy) and observations are size-invariant
(mean-pooled neighbor embedding, see torchrl_swarm/sensing.py) -- the same checkpoint
loads and runs unchanged at any N.

Isaac Sim supports one simulated scene per process, so this script evaluates a single
--num_agents value per invocation. Run it once per swarm size (appending to the same
--results_csv) to build the full N-vs-metrics table, e.g.:

  for n in 5 10 15 20; do
    python scripts/torchrl/eval_formation_scalability.py \\
      --checkpoint logs/torchrl/formation/.../checkpoint.pt \\
      --num_agents $n --results_csv results/formation_scalability.csv
  done

Reports, per swarm size: success rate (all agents reached their assigned slot),
mean time-to-form (steps), mean formation error (final mean distance to assigned slot),
path efficiency (Hungarian-assigned straight-line distance / actual distance traveled --
1.0 is perfectly efficient), and collision rate.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-shot scalability eval for the formation-assignment task.")
parser.add_argument("--task", type=str, default="Formation-TorchRL-UAVSwarm-Direct-v0")
parser.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg_local.yaml")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a MAPPO policy checkpoint (.pt).")
parser.add_argument(
    "--num_agents", type=int, default=5,
    help="Swarm size to evaluate. The checkpoint was trained at 5; pass a larger value to test zero-shot scaling.",
)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--controller", type=str, default="geometric", choices=["geometric", "pd_velocity", "direct"],
)
parser.add_argument("--stage", type=int, default=None, help="Curriculum stage 1-6. Overrides the task's default.")
parser.add_argument("--results_csv", type=str, default=None, help="Append a summary row to this CSV (created if missing).")
parser.add_argument(
    "--stochastic", action="store_true", default=False,
    help="Sample actions (matches how SyncDataCollector actually rolled out during "
         "training) instead of using the deterministic mean action.",
)
parser.add_argument(
    "--residual_rl", action="store_true", default=False,
    help="Checkpoint was trained as a residual correction on top of a P-controller/APF "
         "baseline (see mappo_train.py) -- must be set to reconstruct the same "
         "baseline+scale*policy env-facing action, or playback shows only the small "
         "raw correction term instead of the actual trained behavior.",
)
parser.add_argument("--residual_baseline", type=str, default="point", choices=["point", "apf"])
parser.add_argument("--residual_kp", type=float, default=2.0)
parser.add_argument("--residual_scale", type=float, default=0.3)
parser.add_argument("--residual_repel_gain", type=float, default=0.5)
parser.add_argument(
    "--min_safe_distance", type=float, default=None,
    help="Overrides swarm_cfg.min_safe_distance (R_nh) -- must match the value the "
         "checkpoint was trained with for the APF baseline/containment radius to match.",
)
parser.add_argument("--gravity_radius", type=float, default=None, help="Overrides curriculum.stage8_gravity_radius (R_gv).")
parser.add_argument(
    "--video", action="store_true", default=False,
    help="Record a single continuous video of the whole eval run (step 0 through "
         "--num_steps) via Isaac Sim's built-in camera + gym.wrappers.RecordVideo, same "
         "mechanism as mappo_train.py's --video. Saved under --video_dir.",
)
parser.add_argument("--video_dir", type=str, default="videos/eval", help="Output directory for --video.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

import sys  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import os

import gymnasium as gym
import torch
import yaml
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.modules import ProbabilisticActor, TanhNormal

from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

import UavSwarm.tasks  # noqa: F401

from torchrl_wrapper import IsaacLabTorchRLWrapper
# mappo_train.py is a script with top-level argparse/AppLauncher side effects (it launches
# its own Isaac Sim app on import), so it cannot be imported here. MAPPOPolicy lives in the
# side-effect-free mappo_torchl.py; load_config/make_policy are small enough to duplicate.
from mappo_torchl import MAPPOPolicy


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_policy(obs_dim: int, action_dim: int, config: dict, device: torch.device) -> ProbabilisticActor:
    """Create the shared policy: TensorDictModule -> ProbabilisticActor.

    Mirrors mappo_train.py::make_policy exactly (obs_dim/action_dim-only, N-agnostic).
    """
    net = MAPPOPolicy(obs_dim, action_dim, config["models"]["policy"]["hidden_sizes"]).to(device)
    module = TensorDictModule(
        module=net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    return ProbabilisticActor(
        module=module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )


def _compute_camera_pose(spawn_centroid: torch.Tensor, target: torch.Tensor) -> tuple[list, list]:
    """Elevated 3/4 view of the spawn->target line, close enough that the (9cm)
    Crazyflies are actually visible rather than distant specks, framed automatically
    from the actual episode geometry instead of hand-tuned per --seed.

    eye = midpoint + (perpendicular horizontal offset) + (elevation), lookat = midpoint.
    Offset/elevation scale with the spawn-target distance so framing stays reasonable
    whether that episode's target landed close or far.
    """
    midpoint = (spawn_centroid + target) / 2
    diff = (target - spawn_centroid)[:2]
    dist = diff.norm().clamp(min=1e-3)
    perp = torch.stack([-diff[1], diff[0]]) / dist  # unit horizontal perpendicular
    # Floors sized so the containment sphere (~2m radius / 4m diameter) reads as a
    # clearly-visible but non-dominating landmark rather than filling the frame --
    # confirmed by extracting preview frames at the original (6.0, 3.0) floors, which
    # made a 4m-diameter sphere occupy most of the frame from that distance.
    offset = max(14.0, 1.3 * dist.item())
    elevation = max(7.0, 0.6 * dist.item())
    eye = midpoint.clone()
    eye[0] += perp[0] * offset
    eye[1] += perp[1] * offset
    eye[2] += elevation
    return eye.tolist(), midpoint.tolist()


def build_eval_cfg(task: str, num_agents: int, num_envs: int, device: str, controller_type: str, stage: int | None):
    """Load the task's own registered cfg class (not hardcoded to Formation -- a checkpoint
    trained on FullTask-TorchRL-UAVSwarm-Direct-v0 has a different single_observation_space
    (32, includes RM one-hot) than Formation's (28), so using the wrong cfg class silently
    mismatches obs_rms/policy shapes), then patch it for a swarm size different from the one
    it trained at.

    possible_agents/action_spaces/observation_spaces/state_space are baked as plain class
    attributes at class-definition time (a configclass constraint: no @property, since
    configclass deep-copies fields via setattr -- see torchrl_swarm_env_cfg.py), so setting
    cfg.num_agents alone does not resize them. All four are rebuilt when num_agents differs
    from the class default.
    """
    cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    if stage is not None:
        cfg.curriculum.active_stage = stage
    if num_agents != cfg.num_agents:
        cfg.num_agents = num_agents
        cfg.possible_agents = [f"robot_{i}" for i in range(num_agents)]
        cfg.action_spaces = {
            f"robot_{i}": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for i in range(num_agents)
        }
        cfg.observation_spaces = {
            f"robot_{i}": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(cfg.single_observation_space,))
            for i in range(num_agents)
        }
        cfg.state_space = num_agents * cfg.single_observation_space
    cfg.controller.type = controller_type
    return cfg


def main():
    config = load_config(args_cli.config)
    device = torch.device(config["env"]["device"])
    torch.manual_seed(args_cli.seed)

    env_cfg = build_eval_cfg(args_cli.task, args_cli.num_agents, args_cli.num_envs, str(device), args_cli.controller, args_cli.stage)
    if args_cli.min_safe_distance is not None:
        env_cfg.swarm_cfg.min_safe_distance = args_cli.min_safe_distance
    if args_cli.gravity_radius is not None:
        env_cfg.curriculum.stage8_gravity_radius = args_cli.gravity_radius
    if args_cli.video:
        # env_cfg.viewer.eye/lookat get their real values right after reset, via
        # unwrapped.sim.set_camera_view -- confirmed that call (not just the pre-
        # construction cfg) actually repositions the offscreen RecordVideo camera, by
        # extracting preview frames. enable_translucency is needed for the containment-
        # sphere marker's opacity to actually alpha-blend in the RTX capture -- off by
        # default (PreviewSurfaceCfg.opacity's own docstring: "only affects appearance
        # during interactive rendering" -- confirmed empirically, sphere rendered fully
        # solid without this).
        env_cfg.sim.render.enable_translucency = True

    print(f"\n{'='*80}")
    print("  Formation Scalability Eval — zero-shot swarm-size generalization")
    print(f"{'='*80}")
    print(f"  Task:       {args_cli.task}")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print(f"  Agents:     {args_cli.num_agents}")
    print(f"  Envs:       {args_cli.num_envs}")
    print(f"  Steps:      {args_cli.num_steps}")
    print(f"{'='*80}\n")

    base_env = gym.make(
        args_cli.task, cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if not isinstance(base_env.unwrapped, DirectMARLEnv):
        raise TypeError(f"Expected DirectMARLEnv, got {type(base_env.unwrapped)}")

    if args_cli.video:
        # step_trigger=lambda s: s == 0 records exactly one clip, starting at step 0,
        # running video_length steps -- the whole eval run, not periodic re-triggering
        # (mappo_train.py's --video_interval re-triggers every N steps during long
        # training; here we just want one continuous take of the swarm converging).
        base_env = gym.wrappers.RecordVideo(
            base_env, args_cli.video_dir,
            step_trigger=lambda s: s == 0,
            video_length=args_cli.num_steps,
            disable_logger=True,
        )

    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=True)
    normalize_obs = "obs_rms" in ckpt
    env = IsaacLabTorchRLWrapper(
        base_env, device=str(device), normalize_obs=normalize_obs,
        residual_rl=args_cli.residual_rl, residual_baseline=args_cli.residual_baseline,
        residual_kp=args_cli.residual_kp, residual_scale=args_cli.residual_scale,
        residual_repel_gain=args_cli.residual_repel_gain,
    )
    if normalize_obs:
        assert env.obs_rms is not None
        env.obs_rms.load_state_dict(ckpt["obs_rms"])
    unwrapped = env.unwrapped_env
    num_envs = unwrapped.num_envs
    num_agents = args_cli.num_agents

    print(f"[INFO] obs={env.obs_dim}  action={env.action_dim}  state={env.state_dim}  agents={env.num_agents}  obs_norm={normalize_obs}\n")

    policy = make_policy(env.obs_dim, env.action_dim, config, device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    tensordict = env.reset()

    def _all_positions() -> torch.Tensor:
        # (num_envs, num_agents, 3)
        return torch.stack([rob.data.root_pos_w for rob in unwrapped._robots], dim=1)

    prev_target = unwrapped._desired_pos_w[0, 0].clone()
    if args_cli.video:
        eye, lookat = _compute_camera_pose(_all_positions()[0].mean(dim=0), prev_target)
        unwrapped.sim.set_camera_view(eye, lookat)

    def _dist_to_assigned_slot() -> torch.Tensor:
        # (num_envs, num_agents)
        return torch.linalg.norm(unwrapped._desired_pos_w - _all_positions(), dim=2)

    # Hungarian-assigned straight-line distance per agent, captured right after reset.
    optimal_dist = _dist_to_assigned_slot().clone()

    prev_pos = _all_positions().clone()
    path_length = torch.zeros(num_envs, num_agents, device=device)
    success_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    collided = torch.zeros(num_envs, dtype=torch.bool, device=device)
    final_formation_error = torch.zeros(num_envs, device=device)
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)

    exploration = ExplorationType.RANDOM if args_cli.stochastic else ExplorationType.DETERMINISTIC
    with set_exploration_type(exploration), torch.no_grad():
        for step in range(args_cli.num_steps):
            tensordict = policy(tensordict)
            tensordict = env.step(tensordict)

            cur_pos = _all_positions()
            active = ~finished
            path_length += torch.linalg.norm(cur_pos - prev_pos, dim=2) * active.unsqueeze(1).float()
            prev_pos = cur_pos.clone()

            if args_cli.video:
                # DirectMARLEnv auto-resets a terminated env on the very next step() call
                # -- detect that (env 0's target jumped) and re-center the camera on the
                # new episode's own spawn/target, so a long multi-episode recording stays
                # well-framed throughout instead of only for episode 1.
                cur_target = unwrapped._desired_pos_w[0, 0]
                if torch.linalg.norm(cur_target - prev_target) > 0.5:
                    eye, lookat = _compute_camera_pose(cur_pos[0].mean(dim=0), cur_target)
                    unwrapped.sim.set_camera_view(eye, lookat)
                prev_target = cur_target.clone()

            if hasattr(unwrapped, "_termination_reasons"):
                newly_collided = unwrapped._termination_reasons["collision"] & active
                collided = collided | newly_collided

                goal_reached = unwrapped._termination_reasons["goal_reached"] & active
                success_step[goal_reached] = step
                final_formation_error[goal_reached] = _dist_to_assigned_slot().mean(dim=1)[goal_reached]

            still_running = active & (~collided) & (success_step < 0)
            if step == args_cli.num_steps - 1:
                final_formation_error[still_running] = _dist_to_assigned_slot().mean(dim=1)[still_running]

            finished = finished | collided | (success_step >= 0)

            tensordict = step_mdp(tensordict)

    success_mask = success_step >= 0
    success_rate = success_mask.float().mean().item()
    mean_time_to_form = success_step[success_mask].float().mean().item() if success_mask.any() else float("nan")
    mean_formation_error = final_formation_error.mean().item()
    path_efficiency = (optimal_dist / path_length.clamp(min=1e-3)).mean().item()
    collision_rate = collided.float().mean().item()

    print(f"\n{'='*80}")
    print(f"  Results — num_agents={num_agents}")
    print(f"{'='*80}")
    print(f"  Success rate:        {success_rate:.3f}")
    print(f"  Mean time-to-form:   {mean_time_to_form:.1f} steps")
    print(f"  Mean formation error:{mean_formation_error:.3f} m")
    print(f"  Path efficiency:     {path_efficiency:.3f}  (1.0 = straight-line optimal)")
    print(f"  Collision rate:      {collision_rate:.3f}")
    print(f"{'='*80}\n")

    if args_cli.results_csv:
        os.makedirs(os.path.dirname(args_cli.results_csv) or ".", exist_ok=True)
        write_header = not os.path.exists(args_cli.results_csv)
        with open(args_cli.results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "num_agents", "checkpoint", "success_rate", "mean_time_to_form_steps",
                    "mean_formation_error_m", "path_efficiency", "collision_rate",
                ])
            writer.writerow([
                num_agents, args_cli.checkpoint, success_rate, mean_time_to_form,
                mean_formation_error, path_efficiency, collision_rate,
            ])
        print(f"[INFO] Appended results row to {args_cli.results_csv}\n")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
