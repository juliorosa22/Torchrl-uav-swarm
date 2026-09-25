"""Switching-policy visual eval: a trained SwarmGravity (stage 8) policy drives each agent
toward the shared target; the moment that agent enters the containment sphere, a trained
PackingSwarm (stage 11) policy takes over that agent.

Runs in the stage-10 SwarmGravityRM env, which already does the per-agent sphere-entry
transition (assigns the agent a packing slot by arrival order and repoints its target to
it). Both policies were trained on the 28-dim obs schema; stage 10 appends one trailing
swarm_rm_phase flag, so both policies get obs[..., :28] and that flag picks the policy.

Example:
  python scripts/torchrl/eval_switching_policy.py --headless --video \
      --gravity_checkpoint <stage8 .pt> --packing_checkpoint <packing .pt> --num_steps 2500
"""

import argparse
import sys
from collections import Counter

from isaaclab.app import AppLauncher

# simulation_app.close() exits without flushing block-buffered stdout, which drops every
# print when output is redirected to a file.
sys.stdout.reconfigure(line_buffering=True)

TASK ="SwarmGravityRM-TorchRL-UAVSwarm-Direct-v0"
POLICY_OBS_DIM = 28

parser = argparse.ArgumentParser(description="Switch between a SwarmGravity and a PackingSwarm policy on sphere entry.")
parser.add_argument("--gravity_checkpoint", type=str, required=True, help="Stage-8 (SwarmGravity) checkpoint.")
parser.add_argument("--packing_checkpoint", type=str, required=True, help="Stage-11 (PackingSwarm) checkpoint.")
parser.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg_local.yaml",
                    help="Only read for models.policy.hidden_sizes (both policies must share it).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=2500, help="Env steps to run (50 steps = 1 s of sim time).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--controller", type=str, default="geometric", choices=["geometric", "pd_velocity", "direct"])
parser.add_argument("--stochastic", action="store_true", default=False,
                    help="Sample actions instead of using the deterministic (mean) action.")
parser.add_argument("--residual_kp", type=float, default=2.0)
parser.add_argument("--residual_scale", type=float, default=0.3)
parser.add_argument("--residual_repel_gain", type=float, default=0.5)
parser.add_argument("--video", action="store_true", default=False,
                    help="Record one continuous multi-episode video of env 0 via Isaac Sim's built-in camera.")
parser.add_argument("--video_dir", type=str, default="videos/switching")
parser.add_argument("--no_closeup", action="store_true", default=False,
                    help="Keep the wide approach framing instead of cutting to a close-up of the sphere once "
                         "the first agent enters it.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import yaml
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.modules import ProbabilisticActor, TanhNormal

from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

import UavSwarm.tasks  # noqa: F401

from torchrl_wrapper import IsaacLabTorchRLWrapper
from mappo_torchl import MAPPOPolicy


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_policy(path: str, action_dim: int, config: dict, device: torch.device) -> ProbabilisticActor:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "obs_rms" in ckpt:
        raise ValueError(f"{path} was trained with --normalize_obs; this script feeds raw observations.")
    net = MAPPOPolicy(POLICY_OBS_DIM, action_dim, config["models"]["policy"]["hidden_sizes"]).to(device)
    module = TensorDictModule(
        module=net, in_keys=[("agents", "observation")], out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy


def wide_camera_pose(spawn_centroid: torch.Tensor, target: torch.Tensor) -> tuple[list, list]:
    """Elevated 3/4 view of the spawn->target line (same framing rule as
    eval_formation_scalability.py::_compute_camera_pose, duplicated because that script
    can't be imported -- it launches its own Isaac Sim app at import time)."""
    midpoint = (spawn_centroid + target) / 2
    diff = (target - spawn_centroid)[:2]
    dist = diff.norm().clamp(min=1e-3)
    perp = torch.stack([-diff[1], diff[0]]) / dist
    offset = max(6.5, 1.1 * dist.item())
    elevation = max(3.5, 0.5 * dist.item())
    eye = midpoint.clone()
    eye[0] += perp[0] * offset
    eye[1] += perp[1] * offset
    eye[2] += elevation
    return eye.tolist(), midpoint.tolist()


def closeup_camera_pose(target: torch.Tensor) -> tuple[list, list]:
    eye = target + torch.tensor([2.4, -2.4, 1.7], device=target.device)
    return eye.tolist(), target.tolist()


def classify_outcome(reasons: dict, e: int) -> str:
    if reasons["goal_reached"][e]:
        return "success (all agents packed)"
    if reasons["inter_agent_collision"][e]:
        return "inter-agent collision"
    if reasons["collision"][e]:
        return "altitude violation"
    if reasons["out_of_bounds"][e]:
        return "out of bounds"
    return "timeout"


def main():
    config = load_config(args_cli.config)
    device = torch.device(config["env"]["device"])
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(TASK, device=str(device), num_envs=args_cli.num_envs)
    env_cfg.controller.type = args_cli.controller
    env_cfg.seed = args_cli.seed
    if args_cli.video:
        # Without this the containment-sphere marker renders fully solid in the RTX capture.
        env_cfg.sim.render.enable_translucency = True

    base_env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if not isinstance(base_env.unwrapped, DirectMARLEnv):
        raise TypeError(f"Expected DirectMARLEnv, got {type(base_env.unwrapped)}")
    if args_cli.video:
        base_env = gym.wrappers.RecordVideo(
            base_env, args_cli.video_dir, step_trigger=lambda s: s == 0, video_length=args_cli.num_steps,
            name_prefix="switch_gravity_packing", disable_logger=True,
        )

    env = IsaacLabTorchRLWrapper(
        base_env, device=str(device), residual_rl=True, residual_baseline="apf",
        residual_kp=args_cli.residual_kp, residual_scale=args_cli.residual_scale,
        residual_repel_gain=args_cli.residual_repel_gain,
    )
    unwrapped = env.unwrapped_env
    if env.obs_dim != POLICY_OBS_DIM + 1:
        raise ValueError(f"Expected stage-10 obs_dim={POLICY_OBS_DIM + 1}, got {env.obs_dim}.")
    num_envs, num_agents = unwrapped.num_envs, env.num_agents
    dt = unwrapped.step_dt

    gravity_policy = load_policy(args_cli.gravity_checkpoint, env.action_dim, config, device)
    packing_policy = load_policy(args_cli.packing_checkpoint, env.action_dim, config, device)

    radius = env_cfg.curriculum.get_containment_radius(num_agents, env_cfg.swarm_cfg.min_safe_distance)
    print(f"\n{'=' * 80}")
    print("  Switching-policy eval: SwarmGravity -> PackingSwarm on sphere entry")
    print(f"{'=' * 80}")
    print(f"  Gravity policy: {args_cli.gravity_checkpoint}")
    print(f"  Packing policy: {args_cli.packing_checkpoint}")
    print(f"  Envs: {num_envs}  Agents: {num_agents}  Steps: {args_cli.num_steps} ({args_cli.num_steps * dt:.0f}s)")
    print(f"  Containment radius: {radius:.3f} m   Actions: {'stochastic' if args_cli.stochastic else 'deterministic'}")
    slots = unwrapped._canonical_packing_slots
    pairwise = torch.cdist(slots, slots)
    pairwise.fill_diagonal_(float("inf"))
    print(f"  Packing slots: distance from center {slots.norm(dim=1).min():.3f}-{slots.norm(dim=1).max():.3f} m, "
          f"min pairwise {pairwise.min():.3f} m (R_nh = {env_cfg.swarm_cfg.min_safe_distance} m)")
    print(f"{'=' * 80}\n")

    def act(policy: ProbabilisticActor, obs: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"agents": TensorDict({"observation": obs}, batch_size=[num_envs])}, batch_size=[num_envs])
        return policy(td)["agents", "action"]

    def all_positions() -> torch.Tensor:
        return torch.stack([rob.data.root_pos_w for rob in unwrapped._robots], dim=1)

    def frame_wide():
        eye, lookat = wide_camera_pose(all_positions()[0].mean(dim=0), unwrapped._shared_target_w[0])
        unwrapped.sim.set_camera_view(eye, lookat)

    tensordict = env.reset()
    if args_cli.video:
        frame_wide()

    prev_phase = unwrapped._swarm_rm_phase.clone()
    outcomes: Counter = Counter()
    ep_index, ep_start, entries, closeup_done = 1, 0, [], False

    exploration = ExplorationType.RANDOM if args_cli.stochastic else ExplorationType.DETERMINISTIC
    with set_exploration_type(exploration), torch.no_grad():
        for step in range(args_cli.num_steps):
            obs = tensordict["agents", "observation"][..., :POLICY_OBS_DIM]
            in_sphere = unwrapped._swarm_rm_phase.bool().unsqueeze(-1)  # (E, A, 1)
            tensordict["agents", "action"] = torch.where(in_sphere, act(packing_policy, obs), act(gravity_policy, obs))
            tensordict = env.step(tensordict)

            phase = unwrapped._swarm_rm_phase
            newly_in = (phase == 1) & (prev_phase == 0)
            for agent in newly_in[0].nonzero().flatten().tolist():
                t = (step - ep_start + 1) * dt
                entries.append(f"agent {agent} @ {t:.1f}s -> slot {int(unwrapped._assigned_slot[0, agent])}")
            if args_cli.video and not args_cli.no_closeup and newly_in[0].any() and not closeup_done:
                eye, lookat = closeup_camera_pose(unwrapped._shared_target_w[0])
                unwrapped.sim.set_camera_view(eye, lookat)
                closeup_done = True
            prev_phase = phase.clone()

            ended = unwrapped._last_terminated | unwrapped._last_timed_out
            if ended.any():
                reasons = unwrapped._termination_reasons
                for e in ended.nonzero().flatten().tolist():
                    outcome = classify_outcome(reasons, e)
                    outcomes[outcome] += 1
                    if e == 0:
                        duration = (step - ep_start + 1) * dt
                        switched = "; ".join(entries) if entries else "no agent entered the sphere"
                        print(f"[episode {ep_index}] steps {ep_start}-{step} ({duration:.1f}s) -> {outcome}\n"
                              f"    switches: {switched}")
                        ep_index += 1
                        ep_start, entries, closeup_done = step + 1, [], False
                        if args_cli.video:
                            frame_wide()

            tensordict = step_mdp(tensordict)

    if ep_start < args_cli.num_steps:
        duration = (args_cli.num_steps - ep_start) * dt
        switched = "; ".join(entries) if entries else "no agent entered the sphere"
        print(f"[episode {ep_index}] steps {ep_start}-{args_cli.num_steps - 1} ({duration:.1f}s) -> unfinished when the run ended\n"
              f"    switches: {switched}")

    total = sum(outcomes.values())
    print(f"\n{'=' * 80}\n  Episodes finished across all envs: {total}")
    for name, count in outcomes.most_common():
        print(f"    {name:32s} {count:4d}  ({count / max(total, 1):.0%})")
    print(f"{'=' * 80}\n")
    if args_cli.video:
        print(f"[INFO] Video written under {args_cli.video_dir}/\n")

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
