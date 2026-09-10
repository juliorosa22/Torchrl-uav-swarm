"""Live matplotlib GUI showing SyntheticSwarmEnv agents flying to their goals.

Loads a checkpoint trained by sanity_train.py and runs it deterministically (mode of the
policy's distribution, not sampled) in a single environment so the agent-to-goal behavior
is directly visible. With no --checkpoint given, runs random actions instead (useful to
see what an untrained policy looks like, for contrast).

Usage:
    python scripts/torchrl/visualize_synthetic_env.py
    python scripts/torchrl/visualize_synthetic_env.py --checkpoint <path.pt> --num_agents 5
"""

import argparse
import glob
import os
import torch
import yaml
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from synthetic_swarm_env import SyntheticSwarmEnv
from synthetic_formation_env import SyntheticFormationEnv
from mappo_torchl import MAPPOPolicy
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_latest_checkpoint() -> str | None:
    candidates = glob.glob("logs/torchrl/sanity_check/**/checkpoints/*.pt", recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def make_policy(obs_dim: int, action_dim: int, hidden_sizes, device) -> ProbabilisticActor:
    net = MAPPOPolicy(obs_dim, action_dim, hidden_sizes).to(device)
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


def main():
    parser = argparse.ArgumentParser(description="Visualize SyntheticSwarmEnv agent behavior.")
    parser.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg_sanity.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Policy checkpoint from sanity_train.py. Defaults to the most "
                              "recent one under logs/torchrl/sanity_check/. Pass --random to "
                              "skip loading a checkpoint entirely.")
    parser.add_argument("--random", action="store_true", default=False,
                         help="Ignore any checkpoint and just show untrained/random actions.")
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--trail_len", type=int, default=40, help="Trail length in frames.")
    parser.add_argument(
        "--formation", action="store_true", default=False,
        help="Use SyntheticFormationEnv (shared V-formation slots) instead of "
             "independent per-agent random goals -- matches sanity_train.py --formation.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    num_agents = args.num_agents or config["env"]["num_agents"]
    device = torch.device("cpu")  # single env, tiny nets -- no reason to touch the GPU

    env_cls = SyntheticFormationEnv if args.formation else SyntheticSwarmEnv
    env_kwargs = dict(
        num_envs=1,
        num_agents=num_agents,
        device="cpu",
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

    checkpoint_path = None if args.random else (args.checkpoint or find_latest_checkpoint())
    policy = None
    if checkpoint_path:
        print(f"[INFO] Loading policy from {checkpoint_path}")
        policy = make_policy(env.obs_dim, env.action_dim, config["models"]["policy"]["hidden_sizes"], device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
    else:
        print("[INFO] No checkpoint -- showing random actions.")

    td = env.reset()

    # -- plotting setup ---------------------------------------------------------
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_agents)]
    bound = config["env"]["bound"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-bound * 1.3, bound * 1.3)
    ax.set_ylim(-bound * 1.3, bound * 1.3)
    ax.set_aspect("equal")
    ax.set_title("SyntheticSwarmEnv" + (" (trained policy)" if policy else " (random actions)"))

    empty = torch.zeros(num_agents, 2).numpy()
    agent_scatter = ax.scatter(empty[:, 0], empty[:, 1], s=120, c=colors, zorder=5, edgecolors="black", linewidths=0.8)
    goal_scatter = ax.scatter(empty[:, 0], empty[:, 1], s=200, c=colors, marker="*", zorder=4, edgecolors="black", linewidths=0.5)
    trails = [ax.plot([], [], color=colors[i], alpha=0.5, linewidth=1.5)[0] for i in range(num_agents)]
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10, family="monospace")

    state = {"episode": 1, "step": 0, "trail": [[] for _ in range(num_agents)]}

    def step_env():
        nonlocal td
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            if policy is not None:
                td = policy(td)
            else:
                td["agents", "action"] = torch.rand(1, num_agents, env.action_dim) * 2 - 1
            td = env.step(td)
        done = bool(td["next", "done"].item())
        state["step"] += 1
        td = td["next"]
        if done:
            success = bool(td["terminated"].item())
            print(f"[Episode {state['episode']}] {'SUCCESS' if success else 'TIMEOUT'} after {state['step']} steps")
            state["episode"] += 1
            state["step"] = 0
            state["trail"] = [[] for _ in range(num_agents)]

    def update(_frame):
        step_env()
        pos = env.pos[0].detach().numpy()  # (num_agents, 2)
        goal = env.goal[0].detach().numpy()

        for i in range(num_agents):
            state["trail"][i].append(pos[i].copy())
            if len(state["trail"][i]) > args.trail_len:
                state["trail"][i].pop(0)
            pts = state["trail"][i]
            trails[i].set_data([p[0] for p in pts], [p[1] for p in pts])

        agent_scatter.set_offsets(pos)
        goal_scatter.set_offsets(goal)
        dist = ((goal - pos) ** 2).sum(-1) ** 0.5
        info_text.set_text(
            f"episode {state['episode']}  step {state['step']}/{env.max_episode_steps}\n"
            f"mean dist to goal: {dist.mean():.2f}m"
        )
        return agent_scatter, goal_scatter, *trails, info_text

    interval_ms = 1000.0 / args.fps
    anim = animation.FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
