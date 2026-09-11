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
parser.add_argument("--residual_kp", type=float, default=2.0, help="residual_rl: baseline P-controller gain.")
parser.add_argument("--residual_scale", type=float, default=0.3, help="residual_rl: policy correction weight.")
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

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

from torchrl_wrapper import IsaacLabTorchRLWrapper
from mappo_torchl import MAPPOPolicy, CentralizedCritic, MAPPO
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


def make_critic(state_dim: int, config: dict, device: torch.device) -> TensorDictModule:
    """Create centralized critic.

    Reads ("state",) (concatenated all-agent observations),
    writes ("state_value",).
    """
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
    normalize_obs = args_cli.normalize_obs or config["algorithm"].get("normalize_observations", False)
    print(f"  Obs norm:   {'on' if normalize_obs else 'off'}")
    print(f"  Entropy:    {config['algorithm']['entropy_coef']}")
    if args_cli.residual_rl:
        print(f"  Residual RL: ON  (kp={args_cli.residual_kp}, scale={args_cli.residual_scale})")
    print(f"  Device:     {device}")
    print(f"  Seed:       {config['seed']}")
    print(f"  Envs:       {env_cfg.scene.num_envs}")
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
        residual_rl=args_cli.residual_rl, residual_kp=args_cli.residual_kp,
        residual_scale=args_cli.residual_scale,
    )

    # --- dimensions ---
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    state_dim = env.state_dim
    n_agents = env.num_agents

    print(f"[INFO] obs={obs_dim}  action={action_dim}  state={state_dim}  agents={n_agents}\n")

    # --- create networks ---
    policy = make_policy(obs_dim, action_dim, config, device)
    critic = make_critic(state_dim, config, device)

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
