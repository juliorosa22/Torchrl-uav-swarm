# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script for torchrl_swarm environments with zero-action agent.

Verifies that both FullTask and Baseline variants can be created, reset,
and stepped through without crashes. Also validates observation dimensions
match the expected include_rm_in_obs configuration.

Usage:
  # Test FullTask (23-dim obs, includes RM one-hot):
  python source/UavSwarm/UavSwarm/tasks/direct/torchrl_swarm/test_env.py --task FullTask-TorchRL-UAVSwarm-Direct-v0

  # Test Baseline (19-dim obs, no RM one-hot):
  python source/UavSwarm/UavSwarm/tasks/direct/torchrl_swarm/test_env.py --task Baseline-TorchRL-UAVSwarm-Direct-v0

  # With fewer envs and a step limit:
  python source/UavSwarm/UavSwarm/tasks/direct/torchrl_swarm/test_env.py --task FullTask-TorchRL-UAVSwarm-Direct-v0 --num_envs 4 --num_steps 200
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test torchrl_swarm environment with zero actions.")
parser.add_argument("--task", type=str, default="FullTask-TorchRL-UAVSwarm-Direct-v0",
                    help="Name of the task (FullTask-TorchRL-UAVSwarm-Direct-v0 or Baseline-TorchRL-UAVSwarm-Direct-v0).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to run.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import time

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import DirectMARLEnv

import UavSwarm.tasks  # noqa: F401


def main():
    print("\n" + "=" * 80)
    print(f"  TorchRL Swarm Env Test — {args_cli.task}")
    print("=" * 80)

    # Build config via parse_env_cfg (required by Isaac Lab gym entries)
    print(f"\n[1] Creating environment: {args_cli.task}")
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    is_marl = isinstance(env.unwrapped, DirectMARLEnv)
    if not is_marl:
        print("[ERROR] Expected DirectMARLEnv, aborting.")
        env.close()
        return

    unwrapped = env.unwrapped
    cfg = unwrapped.cfg

    # Report config
    print(f"\n[2] Environment info:")
    print(f"    Class:         {type(unwrapped).__name__}")
    print(f"    Stage:         {cfg.curriculum.active_stage}")
    print(f"    Agents:        {cfg.num_agents}")
    print(f"    Parallel envs: {unwrapped.num_envs}")
    print(f"    Obs dim:       {cfg.single_observation_space}")
    print(f"    Action dim:    {cfg.single_action_space}")
    print(f"    State dim:     {cfg.state_space}")
    print(f"    RM in obs:     {cfg.include_rm_in_obs}")
    print(f"    Episode len:   {cfg.curriculum.get_episode_length()}s")

    # Reset
    print(f"\n[3] Resetting environment...")
    obs_dict = env.reset()[0]
    for agent_name in list(obs_dict.keys())[:2]:
        print(f"    {agent_name}: shape={obs_dict[agent_name].shape}, "
              f"mean={obs_dict[agent_name].mean().item():.4f}, "
              f"min={obs_dict[agent_name].min().item():.4f}, "
              f"max={obs_dict[agent_name].max().item():.4f}")

    expected_obs = cfg.single_observation_space
    actual_obs = obs_dict[cfg.possible_agents[0]].shape[-1]
    if actual_obs == expected_obs:
        print(f"    Obs dimension OK: {actual_obs} == {expected_obs}")
    else:
        print(f"    [WARN] Obs dimension mismatch: got {actual_obs}, expected {expected_obs}")

    # Test _get_states
    print(f"\n[4] Testing centralized state...")
    state = unwrapped._get_states()
    print(f"    State shape:  {state.shape}")
    expected_state = cfg.state_space
    actual_state = state.shape[-1]
    if actual_state == expected_state:
        print(f"    State dimension OK: {actual_state} == {expected_state}")
    else:
        print(f"    [WARN] State dimension mismatch: got {actual_state}, expected {expected_state}")

    # Run zero-action steps
    print(f"\n[5] Running {args_cli.num_steps} steps with zero actions...")
    t_start = time.time()
    total_reward = 0.0
    episodes_done = 0
    num_envs = unwrapped.num_envs
    action_dim = cfg.single_action_space
    zero_actions = {
        agent: torch.zeros(num_envs, action_dim, device=unwrapped.device)
        for agent in cfg.possible_agents
    }

    for step in range(args_cli.num_steps):
        obs_dict, rewards_dict, terminated_dict, truncated_dict, _info = env.step(zero_actions)

        # Accumulate mean reward
        for agent_name in cfg.possible_agents:
            total_reward += rewards_dict[agent_name].mean().item()

        # Count done episodes
        episodes_done += terminated_dict[cfg.possible_agents[0]].sum().item()

        if (step + 1) % 100 == 0:
            elapsed = time.time() - t_start
            avg_reward = total_reward / ((step + 1) * cfg.num_agents)
            print(f"    Step {step + 1:4d}/{args_cli.num_steps} | "
                  f"Avg reward: {avg_reward:+.4f} | "
                  f"Episodes done: {episodes_done:4d} | "
                  f"FPS: {(step + 1) * num_envs / elapsed:.0f}")

    elapsed = time.time() - t_start
    print(f"\n[6] Results:")
    print(f"    Total steps:     {args_cli.num_steps}")
    print(f"    Wall time:       {elapsed:.1f}s")
    print(f"    Avg FPS:         {args_cli.num_steps * num_envs / elapsed:.0f}")
    print(f"    Episodes done:   {episodes_done}")
    avg_reward = total_reward / (args_cli.num_steps * cfg.num_agents)
    print(f"    Avg reward/agt:  {avg_reward:+.4f}")

    # Check RM states
    if hasattr(unwrapped, '_rm_states'):
        rm = unwrapped._rm_states.float()
        print(f"    RM state dist:   H={((rm == 0).sum() / rm.numel() * 100):.0f}%  "
              f"S={((rm == 1).sum() / rm.numel() * 100):.0f}%  "
              f"C={((rm == 2).sum() / rm.numel() * 100):.0f}%  "
              f"O={((rm == 3).sum() / rm.numel() * 100):.0f}%")

    print("\n" + "=" * 80)
    print("  Test passed — environment is working correctly.")
    print("=" * 80 + "\n")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
