"""Shared-weight MAPPO sanity check on a fully synthetic multi-agent env -- no Isaac Sim.

Purpose: the Formation-TorchRL-UAVSwarm task has never converged (goal_reached stuck at
0% across every fix tried -- see the formation-convergence-investigation notes). Before
chasing more UAV-task-specific theories, this script asks a narrower question: can
mappo_torchl.py's MAPPO trainer (GAE, PPO update, shared policy + centralized critic,
individual per-agent reward) converge at all on a trivial problem? SyntheticSwarmEnv
(synthetic_swarm_env.py) is point-mass agents each navigating to their own random goal --
no physics engine, no Isaac Sim, same TensorDict contract as the real task's
IsaacLabTorchRLWrapper (individual reward, state = concatenated per-agent obs) so a
convergence result here is a statement about the pipeline, not about a different problem.

Usage:
    python scripts/torchrl/sanity_train.py --max_iterations 200
    python scripts/torchrl/sanity_train.py --config scripts/torchrl/torchrl_mappo_cfg_sanity.yaml
"""

import argparse
import os
import torch
import yaml
from datetime import datetime
from torchrl.modules import ProbabilisticActor, TanhNormal
from tensordict.nn import TensorDictModule

from synthetic_swarm_env import SyntheticSwarmEnv
from synthetic_formation_env import SyntheticFormationEnv
from mappo_torchl import MAPPOPolicy, CentralizedCritic, MAPPO


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_policy(obs_dim: int, action_dim: int, config: dict, device) -> ProbabilisticActor:
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


def make_critic(state_dim: int, config: dict, device) -> TensorDictModule:
    net = CentralizedCritic(state_dim, config["models"]["critic"]["hidden_sizes"]).to(device)
    return TensorDictModule(module=net, in_keys=[("state",)], out_keys=[("state_value",)])


def main():
    parser = argparse.ArgumentParser(description="Synthetic-env MAPPO sanity check (no Isaac Sim).")
    parser.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg_sanity.yaml")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="mappo_sanity")
    parser.add_argument("--experiment_directory", type=str, default=None)
    parser.add_argument(
        "--formation", action="store_true", default=False,
        help="Use SyntheticFormationEnv (shared, Hungarian-assigned V-formation slots) "
             "instead of independent per-agent random goals -- isolates whether the "
             "formation/coordination structure itself (not UAV flight dynamics) blocks "
             "convergence on the real Formation-TorchRL-UAVSwarm task.",
    )
    args = parser.parse_args()

    if args.formation and args.model_name == "mappo_sanity":
        args.model_name = "mappo_sanity_formation"

    config = load_config(args.config)

    num_envs = args.num_envs or config["env"]["num_envs"]
    num_agents = args.num_agents or config["env"]["num_agents"]
    device = torch.device(args.device or config["env"]["device"])
    seed = args.seed if args.seed is not None else config["seed"]
    torch.manual_seed(seed)

    if args.experiment_directory:
        config["training"]["experiment_directory"] = args.experiment_directory
    log_root = os.path.abspath(os.path.join("logs", "torchrl", config["training"]["experiment_directory"]))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.model_name}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  MAPPO Sanity Check -- Synthetic Point-Mass Env (no Isaac Sim)")
    print(f"{'='*80}")
    print(f"  Envs:   {num_envs}   Agents: {num_agents}   Device: {device}   Seed: {seed}")
    print(f"  Log:    {log_dir}")
    print(f"{'='*80}\n")

    env_cls = SyntheticFormationEnv if args.formation else SyntheticSwarmEnv
    env_kwargs = dict(
        num_envs=num_envs,
        num_agents=num_agents,
        device=str(device),
        max_episode_steps=config["env"]["max_episode_steps"],
        max_speed=config["env"]["max_speed"],
        dt=config["env"]["dt"],
        bound=config["env"]["bound"],
        goal_threshold=config["env"]["goal_threshold"],
        goal_bonus=config["env"]["goal_bonus"],
    )
    if args.formation:
        env_kwargs["formation_spacing"] = config["env"].get("formation_spacing", 1.0)
    env = env_cls(**env_kwargs)

    obs_dim, action_dim, state_dim = env.obs_dim, env.action_dim, env.state_dim
    policy = make_policy(obs_dim, action_dim, config, device)
    critic = make_critic(state_dim, config, device)

    n_policy = sum(p.numel() for p in policy.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"[INFO] Policy params: {n_policy:,}  |  Critic params: {n_critic:,}\n")

    frames_per_batch = config["algorithm"]["frames_per_batch"]
    total_frames = args.max_iterations * frames_per_batch if args.max_iterations else config["training"]["total_frames"]

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
        n_agents=num_agents,
        frames_per_batch=frames_per_batch,
        model_name=args.model_name,
        log_dir=log_dir,
        checkpoint_interval=config["training"]["checkpoint_interval"],
        normalize_advantage=config["algorithm"].get("normalize_advantages", True),
        normalize_rewards=config["algorithm"].get("normalize_rewards", True),
        max_grad_norm=config["algorithm"].get("max_grad_norm", 1.0),
    )

    print(f"\n[INFO] Starting training -- {total_frames:,} frames\n")
    trainer.train(total_frames)
    print("\n[INFO] Done.\n")
    env.close()


if __name__ == "__main__":
    main()
