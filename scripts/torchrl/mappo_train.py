# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shared-weight MAPPO training with TorchRL for Isaac Lab UAV swarm.

Single policy network processes all agents. Centralized critic sees
concatenated state. Agent dimension is flattened into batch dimension
during PPO updates so gradients from all agents flow into shared weights.
"""

import argparse
import sys
import torch
import yaml
import gymnasium as gym
from torchrl.modules import ProbabilisticActor, TanhNormal
from tensordict.nn import TensorDictModule
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Shared-weight MAPPO training with TorchRL.")
parser.add_argument("--task", type=str, default="FullTask-TorchRL-UAVSwarm-Direct-v0")
parser.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg_local.yaml")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument(
    "--num_agents", type=int, default=None,
    help="Overrides env_cfg.num_agents (default 5). possible_agents/action_spaces/"
         "observation_spaces/state_space are baked at class-definition time from "
         "num_agents (a @configclass constraint), so all four are rebuilt here -- "
         "same pattern as eval_formation_scalability.py's build_eval_cfg.",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--model_name", type=str, default="mappo_uav_swarm")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument(
    "--controller", type=str, default="geometric",
    choices=["geometric", "pd_velocity", "direct"],
    help="Low-level controller: geometric=SE(3) (default), pd_velocity=approx PD, direct=original force/torque.",
)
parser.add_argument("--stage", type=int, default=None, help="Curriculum stage 1-5. Overrides config.")
parser.add_argument(
    "--simple_reward", action="store_true", default=False,
    help="Stage 6 only: use get_formation_rewards_simple() (pose-distance term only, no "
         "delta/alignment/safety/jerk terms) instead of the full formation reward. Diagnostic "
         "switch -- see CurriculumCfg.stage6_simple_reward in torchrl_swarm_env_cfg.py.",
)
parser.add_argument(
    "--normalize_obs", action="store_true", default=False,
    help="Running per-dimension mean/std normalization on the observation vector "
         "(IsaacLabTorchRLWrapper's obs_rms). Off by default -- see algorithm."
         "normalize_observations in the config yaml.",
)
parser.add_argument(
    "--residual_rl", action="store_true", default=False,
    help="Env receives compute_baseline_action(pos, desired_pos_w) + residual_scale * "
         "policy_action instead of the raw policy action -- PPO learns a correction on "
         "top of a proven P-controller (see controller.py) instead of the full command "
         "from scratch. See formation-convergence-investigation memory for why.",
)
parser.add_argument(
    "--residual_baseline", type=str, default="point", choices=["point", "apf"],
    help="residual_rl: 'point' = plain P-controller toward desired_pos_w; 'apf' = adds "
         "inter-agent repulsion for the swarm-gravity task (SwarmGravity-TorchRL-UAVSwarm-Direct-v0).",
)
parser.add_argument(
    "--min_safe_distance", type=float, default=None,
    help="Overrides swarm_cfg.min_safe_distance (R_nh) -- the soft inter-agent avoidance "
         "radius used by the swarm-gravity reward/APF baseline and stage 8's derived "
         "containment radius. Default (config): 1.0m.",
)
parser.add_argument(
    "--gravity_radius", type=float, default=None,
    help="Overrides curriculum.stage8_gravity_radius (R_gv) -- how close the nearest "
         "agent must get to the shared target to count as arrived. Default (config): 0.5m.",
)
parser.add_argument(
    "--packing_density", type=float, default=None,
    help="Overrides curriculum.stage8_packing_density used by the derived containment-"
         "sphere radius. Default (config): 0.6.",
)
parser.add_argument(
    "--stage8_episode_length", type=float, default=None,
    help="Overrides curriculum.stage8_episode_length_s (default 120.0). The containment "
         "condition (all agents inside R_containment, closest within R_gv) requires the "
         "whole swarm to settle into a tight near-packing-limit arrangement, not just "
         "approach a point -- time_out was 12.6%% of episodes in the R_nh=2.0/200k-frame "
         "run's last third, second only to goal_reached itself, suggesting some episodes "
         "are still settling when the clock runs out.",
)
parser.add_argument(
    "--frames_per_batch", type=int, default=None,
    help="Overrides algorithm.frames_per_batch. Rollout length per env, T = "
         "frames_per_batch/num_envs, is what actually matters for GAE bootstrapping -- "
         "the remote profile's default (32768/256=128) is 4x shorter than local's "
         "(8192/16=512), a live suspect for the 256-env regression (see "
         "formation-convergence-investigation memory). E.g. 131072 preserves T=512 at "
         "num_envs=256.",
)
parser.add_argument("--residual_kp", type=float, default=2.0, help="residual_rl: baseline attraction gain.")
parser.add_argument("--residual_scale", type=float, default=0.3, help="residual_rl: policy correction weight.")
parser.add_argument("--residual_repel_gain", type=float, default=0.5, help="residual_rl: apf baseline's repulsion strength.")
parser.add_argument(
    "--experiment_directory", type=str, default=None,
    help="Base folder under logs/torchrl/ shared by every run of this experiment; each run "
         "still gets its own timestamped subdir inside it. Overrides the config file's value.",
)
parser.add_argument(
    "--entropy_coef", type=float, default=None,
    help="Overrides config algorithm.entropy_coef (default 0.01). Diagnostic switch for "
         "premature entropy collapse (see formation-convergence-investigation memory).",
)
parser.add_argument(
    "--target_kl", type=float, default=None,
    help="Standard PPO early-stopping (Spinning Up/CleanRL default ~0.01-0.02): once an "
         "iteration's epoch-mean approximate KL exceeds this, stop taking further gradient "
         "steps that iteration instead of blindly running the full n_epochs. None (default) "
         "disables it -- see MAPPO.__init__'s target_kl docstring for why this was added "
         "(swarmgravity-rm-paper memory: a runaway policy update with no other guard).",
)
parser.add_argument(
    "--disable_reset_jitter", action="store_true", default=False,
    help="Diagnostic-only: reproduces the old behavior where only the initial full-batch "
         "reset staggers episode_length_buf phase, not every ordinary per-env reset. "
         "For the A/B multi-seed comparison testing the 'staggered resets' hypothesis "
         "(see formation-convergence-investigation memory / arXiv:2511.21011) -- default "
         "(flag absent) keeps the fix on.",
)
parser.add_argument(
    "--critic_arch", type=str, default="flat", choices=["flat", "attention"],
    help="'flat' (default): plain MLP over the concatenated-obs state (CentralizedCritic). "
         "'attention': self-attention over per-agent embeddings, biased by the true "
         "pairwise agent distance (GraphAttentionCritic) -- permutation-invariant and "
         "N-agnostic, unlike 'flat'. Requires a task whose state appends a trailing "
         "NxN distance matrix after the per-agent obs concat (e.g. SwarmGravityV2 via "
         "include_distance_matrix_in_state); will error on a task that doesn't.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

from torchrl_wrapper import IsaacLabTorchRLWrapper
from mappo_torchl import MAPPOPolicy, CentralizedCritic, GraphAttentionCritic, MAPPO
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.utils.io import dump_yaml
from datetime import datetime

import UavSwarm.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_policy(obs_dim: int, action_dim: int, config: dict, device: torch.device) -> ProbabilisticActor:
    """Create shared policy: TensorDictModule → ProbabilisticActor.

    Reads ("agents", "observation"), writes ("agents", "action").
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


def make_critic(
    state_dim: int, config: dict, device: torch.device,
    critic_arch: str = "flat", num_agents: int | None = None, obs_dim: int | None = None,
) -> TensorDictModule:
    """Create centralized critic.

    Reads ("state",) (concatenated all-agent observations, optionally followed by extra
    privileged dims a task appends -- see torchrl_swarm_env.py::_get_states), writes
    ("state_value",). "attention" requires num_agents/obs_dim (to split state into
    per-agent node features vs. the trailing pairwise-distance matrix) -- only tasks that
    set include_distance_matrix_in_state (e.g. SwarmGravityV2) provide that trailing part.
    """
    if critic_arch == "attention":
        if num_agents is None or obs_dim is None:
            raise ValueError("--critic_arch attention requires num_agents and obs_dim.")
        net = GraphAttentionCritic(num_agents, obs_dim).to(device)
    else:
        net = CentralizedCritic(state_dim, config["models"]["critic"]["hidden_sizes"]).to(device)
    return TensorDictModule(
        module=net,
        in_keys=[("state",)],
        out_keys=[("state_value",)],
    )


agent_cfg_entry_point = "skrl_mappo_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: DirectMARLEnvCfg, agent_cfg: dict):
    config = load_config(args_cli.config)

    # --- overrides ---
    env_cfg.scene.num_envs = args_cli.num_envs or config["env"]["num_envs"]
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    if args_cli.seed is not None:
        config["seed"] = args_cli.seed
    torch.manual_seed(config["seed"])
    env_cfg.seed = config["seed"]

    # --- controller and curriculum overrides ---
    # Apply YAML gains first, then CLI type takes final precedence.
    if not hasattr(env_cfg, "controller"):
        raise AttributeError(
            f"env_cfg ({type(env_cfg).__name__}) has no 'controller' attribute. "
            "Use a TorchRL task: 'FullTask-TorchRL-UAVSwarm-Direct-v0' or 'Baseline-TorchRL-UAVSwarm-Direct-v0'."
        )
    if "controller" in config:
        for k, v in config["controller"].items():
            setattr(env_cfg.controller, k, v)
    env_cfg.controller.type = args_cli.controller
    if args_cli.stage is not None:
        env_cfg.curriculum.active_stage = args_cli.stage
    if args_cli.simple_reward:
        env_cfg.curriculum.stage6_simple_reward = True
    if args_cli.entropy_coef is not None:
        config["algorithm"]["entropy_coef"] = args_cli.entropy_coef
    if args_cli.target_kl is not None:
        config["algorithm"]["target_kl"] = args_cli.target_kl
    if args_cli.frames_per_batch is not None:
        config["algorithm"]["frames_per_batch"] = args_cli.frames_per_batch
    if args_cli.min_safe_distance is not None:
        env_cfg.swarm_cfg.min_safe_distance = args_cli.min_safe_distance
    if args_cli.gravity_radius is not None:
        env_cfg.curriculum.stage8_gravity_radius = args_cli.gravity_radius
    if args_cli.stage8_episode_length is not None:
        env_cfg.curriculum.stage8_episode_length_s = args_cli.stage8_episode_length
    if args_cli.disable_reset_jitter:
        env_cfg.disable_reset_jitter = True
    if args_cli.packing_density is not None:
        env_cfg.curriculum.stage8_packing_density = args_cli.packing_density
    if args_cli.num_agents is not None and args_cli.num_agents != env_cfg.num_agents:
        n = args_cli.num_agents
        env_cfg.num_agents = n
        env_cfg.possible_agents = [f"robot_{i}" for i in range(n)]
        env_cfg.action_spaces = {
            f"robot_{i}": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for i in range(n)
        }
        env_cfg.observation_spaces = {
            f"robot_{i}": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(env_cfg.single_observation_space,))
            for i in range(n)
        }
        env_cfg.state_space = n * env_cfg.single_observation_space

    device = torch.device(config["env"]["device"])

    # --- log dir ---
    if args_cli.experiment_directory:
        config["training"]["experiment_directory"] = args_cli.experiment_directory
    log_root = os.path.abspath(os.path.join("logs", "torchrl", config["training"]["experiment_directory"]))
    log_dir = os.path.join(
        log_root,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_mappo_{args_cli.model_name}",
    )
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "torchrl_config.yaml"), config)
    env_cfg.log_dir = log_dir

    print(f"\n{'='*80}")
    print(f"  Shared-Weight MAPPO — TorchRL")
    print(f"{'='*80}")
    print(f"  Task:       {args_cli.task}")
    print(f"  Controller: {env_cfg.controller.type}  (max_vel={env_cfg.controller.max_lin_vel_cmd} m/s)")
    print(f"  Stage:      {env_cfg.curriculum.active_stage}")
    print(f"  Agents:     {env_cfg.num_agents}")
    if env_cfg.curriculum.active_stage == 6:
        print(f"  Reward:     {'simple (pose-distance only)' if env_cfg.curriculum.stage6_simple_reward else 'full formation'}")
    if env_cfg.curriculum.active_stage == 8:
        r_nh = env_cfg.swarm_cfg.min_safe_distance
        r_gv = env_cfg.curriculum.stage8_gravity_radius
        r_containment = env_cfg.curriculum.get_containment_radius(env_cfg.num_agents, r_nh)
        print(f"  R_nh:       {r_nh} m   R_gv: {r_gv} m   R_containment (derived): {r_containment:.3f} m")
        print(f"  Episode length: {env_cfg.curriculum.stage8_episode_length_s} s")
    normalize_obs = args_cli.normalize_obs or config["algorithm"].get("normalize_observations", False)
    print(f"  Obs norm:   {'on' if normalize_obs else 'off'}")
    print(f"  Entropy:    {config['algorithm']['entropy_coef']}")
    target_kl = config["algorithm"].get("target_kl")
    print(f"  Target KL:  {target_kl if target_kl is not None else 'off'}")
    print(f"  Critic:     {args_cli.critic_arch}")
    if args_cli.residual_rl:
        print(f"  Residual RL: ON  (baseline={args_cli.residual_baseline}, kp={args_cli.residual_kp}, scale={args_cli.residual_scale}, repel={args_cli.residual_repel_gain})")
    print(f"  Device:     {device}")
    print(f"  Seed:       {config['seed']}")
    print(f"  Envs:       {env_cfg.scene.num_envs}")
    fpb = config["algorithm"]["frames_per_batch"]
    print(f"  Frames/batch: {fpb}   T (rollout len) = {fpb / env_cfg.scene.num_envs:.0f}")
    print(f"  Log:        {log_dir}")
    print(f"{'='*80}\n")

    # --- create env ---
    base_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if not isinstance(base_env.unwrapped, DirectMARLEnv):
        raise TypeError(f"Expected DirectMARLEnv, got {type(base_env.unwrapped)}")

    if args_cli.video:
        base_env = gym.wrappers.RecordVideo(
            base_env, os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda s: s % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = IsaacLabTorchRLWrapper(
        base_env, device=str(device), normalize_obs=normalize_obs,
        residual_rl=args_cli.residual_rl, residual_baseline=args_cli.residual_baseline,
        residual_kp=args_cli.residual_kp, residual_scale=args_cli.residual_scale,
        residual_repel_gain=args_cli.residual_repel_gain,
    )

    # --- dimensions ---
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    state_dim = env.state_dim
    n_agents = env.num_agents

    print(f"[INFO] obs={obs_dim}  action={action_dim}  state={state_dim}  agents={n_agents}\n")

    # --- create networks ---
    policy = make_policy(obs_dim, action_dim, config, device)
    critic = make_critic(
        state_dim, config, device,
        critic_arch=args_cli.critic_arch, num_agents=n_agents, obs_dim=obs_dim,
    )

    n_policy = sum(p.numel() for p in policy.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"[INFO] Policy params: {n_policy:,}  |  Critic params: {n_critic:,}\n")

    # --- trainer ---
    frames_per_batch = config["algorithm"]["frames_per_batch"]
    if args_cli.max_iterations:
        total_frames = args_cli.max_iterations * frames_per_batch
    else:
        total_frames = config["training"]["total_frames"]

    trainer = MAPPO(
        env=env,
        policy=policy,
        critic=critic,
        device=device,
        lr=config["algorithm"]["learning_rate"],
        gamma=config["algorithm"]["gamma"],
        gae_lambda=config["algorithm"]["gae_lambda"],
        clip_epsilon=config["algorithm"]["clip_epsilon"],
        c1=config["algorithm"]["value_loss_coef"],
        c2=config["algorithm"]["entropy_coef"],
        n_epochs=config["algorithm"]["n_epochs"],
        batch_size=config["algorithm"]["batch_size"],
        n_agents=n_agents,
        frames_per_batch=frames_per_batch,
        model_name=args_cli.model_name,
        log_dir=log_dir,
        checkpoint_interval=config["training"]["checkpoint_interval"],
        normalize_advantage=config["algorithm"].get("normalize_advantages", True),
        normalize_rewards=config["algorithm"].get("normalize_rewards", True),
        max_grad_norm=config["algorithm"].get("max_grad_norm", 1.0),
        target_kl=config["algorithm"].get("target_kl"),
    )

    # --- resume ---
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        ckpt = torch.load(args_cli.checkpoint, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        critic.load_state_dict(ckpt["critic"])
        if normalize_obs and "obs_rms" in ckpt:
            assert env.obs_rms is not None
            env.obs_rms.load_state_dict(ckpt["obs_rms"])
        elif normalize_obs:
            print("[WARN] --normalize_obs is on but the checkpoint has no obs_rms stats "
                  "(it was likely trained without normalization) -- resuming with fresh "
                  "(mean=0, var=1) statistics, which will not match what the loaded policy "
                  "was trained on until they reconverge.")
        # Sync collector with loaded weights
        trainer.collector.update_policy_weights_()

    # --- train ---
    print(f"\n[INFO] Starting training — {total_frames:,} frames\n")
    trainer.train(total_frames)
    print("\n[INFO] Done.\n")
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
