# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent Reinforcement Learning (MARL) for quadrotor swarm control using Crazyflie micro-UAVs in the **IsaacLab + IsaacSim** simulation stack. The goal is training scalable decentralized swarm policies with GPU-accelerated physics for navigation, obstacle avoidance, inter-agent collision avoidance, and formation control.

## Environment / Simulation Stack

- **Isaac Sim 4.5–5.1** with **Isaac Lab** (pre-vectorized GPU simulation, `DirectMARLEnv` workflow)
- **Python >= 3.10**, PyTorch, gymnasium
- RL libraries: **skrl >= 1.4.3** (primary: MAPPO/IPPO) and **TorchRL** (secondary)
- Environments are pre-vectorized — a single `gym.make` call spins up hundreds of parallel envs (default 256) on GPU

## Build / Install

The package is an IsaacLab extension. Install from the extension root:

```bash
cd source/UavSwarm && pip install -e .
```

## Commands

### Training (skrl — primary framework)

```bash
# Baseline environment (MAPPO):
python scripts/skrl/train.py --task=Baseline-UAVSwarm-Direct-v0 --headless --ml_framework torch --algorithm MAPPO

# Full-task environment with shared policy:
python scripts/skrl/train_shared.py --task=FullTask-UAVSwarm-Direct-v0 --headless --ml_framework torch --algorithm MAPPO

# Resume from checkpoint:
python scripts/skrl/train.py --task=Baseline-UAVSwarm-Direct-v0 --headless --checkpoint <path.pt> --ml_framework torch --algorithm MAPPO

# With video recording:
python scripts/skrl/train.py --task=Baseline-UAVSwarm-Direct-v0 --headless --ml_framework torch --algorithm MAPPO --video --video_length 1000 --video_interval 50000 --enable_cameras
```

### Evaluation / Playback

```bash
# Play a trained checkpoint:
python scripts/skrl/play.py --task=Baseline-UAVSwarm-Direct-v0 --checkpoint <path.pt> --algorithm MAPPO --num_envs 32

# Evaluate individual agent policies and clone the best one:
python scripts/skrl/eval_policy.py --task=Baseline-UAVSwarm-Direct-v0 --checkpoint <path.pt> --algorithm MAPPO
```

### Monitoring

```bash
# TensorBoard (must be run from IsaacLab root):
./isaaclab.sh -p -m tensorboard.main --logdir=logs
```

### Linting / Formatting

```bash
# Pre-commit hooks (black, flake8, isort, codespell, etc.):
pre-commit run --all-files

# Flake8 standalone:
flake8 source/
```

- Black line length: 120, flake8 complexity limit: 30
- Uses Google-style docstrings (`docstring-convention=google`)

## Architecture

### Curriculum Stages (5-stage progressive training)

The environment is configured in `CurricumulCfg.active_stage` (1–5):

| Stage | Description | Key Behavior |
|-------|-------------|-------------|
| 1 | Hovering | Stabilize in place |
| 2 | Point-to-Point Nav | Reach targets, no obstacles |
| 3 | Navigation with Static Obstacles | Zig-zag obstacle course with waypoints |
| 4 | Multi-Agent Formation | Maintain inverted-V formation while moving |
| 5 | Swarm + Obstacles | Formation + obstacle avoidance |

Each stage has its own episode length and reset logic. The active stage is set in the config class, not via CLI.

### Environment Class Structure (`DirectMARLEnv` subclass)

The main environment is `BaselineUAVSwarmEnv` in `source/UavSwarm/UavSwarm/tasks/direct/baseline_uavswarm/baseline_uavswarm_env.py`. It follows the Direct MARL workflow:

1. **`_setup_scene`** — Creates N Crazyflie `Articulation` instances and obstacles (stages 3/5), then clones environments via `scene.clone_environments()`
2. **`_get_observations`** — Returns per-agent dict: `{robot_0: (num_envs, 19), ...}`. 19-dim: lin_vel(3) + ang_vel(3) + projected_gravity(3) + desired_pos_b(3) + obstacle_dist(1) + neighbor_vel_b(3) + neighbor_pos_b(3). RM states (4-dim one-hot) are commented out for the baseline.
3. **`_pre_physics_step`** — Converts normalized policy actions `[-1,1]` to thrust/moments, updates RM states and waypoint goals
4. **`_get_rewards`** — Energy-based reward: position energy (inverse quadratic), distance delta, velocity alignment, smoothness multiplier, per-RM-state weighting. Obstacle and cooperation terms active only in stages 3-5.
5. **`_get_dones`** — Collision, out-of-bounds, goal-reached (curriculum-aware), and timeout
6. **`_get_states`** — Centralized critic state: all agent observations concatenated → `(num_envs, num_agents * 19)` for MAPPO centralized critic
7. **`_reset_idx`** — Logs metrics, resets robots, applies curriculum-dependent position sampling

### Caching Optimization

The environment uses a lazy cache (`_cache_valid` flag) to avoid redundant computation. `_ensure_cache_populated()` computes obstacle distances and nearest-neighbor data once per step; `_get_observations()`, `_get_rewards()`, and `_get_states()` all consume the cached values.

### Task Variants

| Task Name | Directory | Description |
|-----------|-----------|-------------|
| `Baseline-UAVSwarm-Direct-v0` | `baseline_uavswarm/` | Stage 5 only, 19-dim obs (no RM states), 5 agents |
| `FullTask-UAVSwarm-Direct-v0` | `fulltask_swarm_rm/` | Full RM integration, 23-dim obs (includes RM one-hot), supports all stages |

Both share the same architecture; the baseline strips RM states from observations and is locked to stage 5.

### Reward Machine States

The environment tracks 4 states per agent (vectorized, GPU):
- **H (0)**: Hovering — agent_z < exit_hover_altitude
- **S (1)**: Single-moving — above hover, far from obstacles and neighbors
- **C (2)**: Coop-moving — above hover, far from obstacles, near neighbor
- **O (3)**: Obstacle-avoiding — above hover, close to obstacle

State transitions happen in `_switch_rm_state()`. In the baseline, RM states affect reward weighting but are not in observations.

### Agent Configuration

MAPPO is the primary algorithm with per-agent policy/value networks (`separate: True`). Configs are in `agents/skrl_mappo_cfg.yaml`:
- Policy: `[128, 128]` with ReLU, GaussianMixin
- Value/Critic: `[256, 256, 128]` with ReLU, DeterministicMixin
- Rollouts: 128, learning epochs: 10, mini-batches: 32
- KL-adaptive LR scheduler, grad norm clip: 0.5
- Target timesteps: 5,000,000

### Key Configuration Parameters

- `num_agents: 5` — Number of Crazyflie drones per environment
- `scene.num_envs: 256` — Parallel environments
- `decimation: 2` — Action repeats (sim dt = 1/100s, effective step = 1/50s)
- `thrust_to_weight: 1.9` — Thrust scaling factor
- `moment_scale: 0.01` — Moment/torque scaling
- `obstacles_size: (0.15, 0.8, 8.0)` — Stage 5 obstacles: thin walls (0.15m thick, 8m tall)

### Code Conventions

- All computation is vectorized across drones and environments using PyTorch tensors (no Python loops in per-step methods)
- Tensor shapes: drone batch is dim 0 `(num_drones, num_envs, ...)`, environment batch is dim 1 after transpose
- Robot data accessed via `robot.data.root_pos_w`, `robot.data.root_lin_vel_b`, etc. (IsaacLab Articulation API)
- Uses `@configclass` decorator from `isaaclab.utils` for all config dataclasses
- License: BSD-3-Clause (Isaac Lab), Apache-2.0 (package)

### Extension Registration

The package is registered as an Isaac Lab extension via `config/extension.toml`. Task registration happens in `UavSwarm/tasks/__init__.py` which uses `isaaclab_tasks.utils.import_packages()` to recursively import and register all config-based Gym environments.
