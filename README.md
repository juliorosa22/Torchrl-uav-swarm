# IsaacLab UAV Swarm Navigation
<p align="center">
  <img src="assets/isaac-sim-env.png" width="80%" alt="UAV Swarm Environment in Isaac Sim">
</p>

## Overview

**Multi-Agent Reinforcement Learning (MARL)** for controlling a **Crazyflie quadrotor swarm** within the **IsaacLab + IsaacSim** GPU-accelerated simulation stack.

The system trains scalable decentralized swarm policies supporting:
- Multi-agent UAV navigation
- Obstacle avoidance
- Inter-agent collision avoidance
- Formation learning
- Curriculum-based progressive training (5 stages)

Two RL frameworks are supported: **skrl** (primary, MAPPO/IPPO) and **TorchRL** (secondary, MAPPO).

---

## Architecture

### Environment Variants

| Task ID | Description | Obs Dim | Agents |
|---------|-------------|---------|--------|
| `Baseline-UAVSwarm-Direct-v0` | Stage 5, no Reward Machine states | 28 | 5 |
| `FullTask-UAVSwarm-Direct-v0` | Full RM integration, all stages | 32 | 5 |
| `Baseline-TorchRL-UAVSwarm-Direct-v0` | TorchRL backend, baseline obs | 28 | 5 |
| `FullTask-TorchRL-UAVSwarm-Direct-v0` | TorchRL backend, full obs | 32 | 5 |

### Observation Space (28-dim baseline)

| Component | Dim | Description |
|-----------|-----|-------------|
| `lin_vel_b` | 3 | Linear velocity in body frame |
| `ang_vel_b` | 3 | Angular velocity in body frame |
| `projected_gravity` | 3 | Gravity vector projected to body frame |
| `desired_pos_b` | 3 | Goal position in body frame |
| `obstacle_dist` | 1 | Distance to nearest obstacle |
| `obstacle_dir_b` | 3 | Bearing to nearest obstacle in body frame |
| `neighbor_vel_b` | 3 | Mean-pooled neighbor velocity (body frame) |
| `neighbor_pos_b` | 3 | Mean-pooled neighbor position (body frame) |
| `neighbor_dist_mean` | 1 | Mean distance to neighbors |
| `neighbor_bearing_b` | 3 | Mean bearing to neighbors (body frame) |

The FullTask variant appends a 4-dim Reward Machine one-hot state.

### Curriculum Stages

| Stage | Description | Key Behavior |
|-------|-------------|--------------|
| 1 | Hovering | Stabilize in place |
| 2 | Point-to-Point Nav | Reach targets, no obstacles |
| 3 | Navigation with Static Obstacles | Zig-zag obstacle course |
| 4 | Multi-Agent Formation | Inverted-V formation while moving |
| 5 | Swarm + Obstacles | Formation + obstacle avoidance |

<p align="center">
  <img src="assets/swarm_progress.gif" width="80%" alt="UAV Swarm Hover Training Demo">
</p>

### SE(3) Geometric Controller

A cascaded SE(3) geometric controller (`GeometricSE3Controller`) sits between the policy outputs and the physics engine. It converts desired accelerations into body-frame thrust and moment commands, enabling smoother and more physically consistent control than raw thrust mapping.

---

## Installation

This package is an IsaacLab extension. Requires **Isaac Sim 4.5+** and **Isaac Lab**.

```bash
# Install the extension (from repo root):
cd source/UavSwarm && pip install -e .
```

Dependencies:
- Python >= 3.10
- PyTorch
- skrl >= 1.4.3
- TorchRL (for TorchRL pipeline)
- gymnasium

---

## Training

### skrl (primary)

```bash
# Baseline environment — MAPPO:
python scripts/skrl/train.py \
  --task=Baseline-UAVSwarm-Direct-v0 \
  --headless --ml_framework torch --algorithm MAPPO

# Full-task environment with shared policy:
python scripts/skrl/train_shared.py \
  --task=FullTask-UAVSwarm-Direct-v0 \
  --headless --ml_framework torch --algorithm MAPPO

# Resume from checkpoint:
python scripts/skrl/train.py \
  --task=Baseline-UAVSwarm-Direct-v0 \
  --headless --checkpoint <path.pt> \
  --ml_framework torch --algorithm MAPPO

# With video recording:
python scripts/skrl/train.py \
  --task=Baseline-UAVSwarm-Direct-v0 \
  --headless --ml_framework torch --algorithm MAPPO \
  --video --video_length 1000 --video_interval 50000 --enable_cameras
```

### TorchRL (secondary)

```bash
python scripts/torchrl/train.py \
  --task=Baseline-TorchRL-UAVSwarm-Direct-v0 --headless
```

---

## Evaluation

```bash
# Play a trained checkpoint:
python scripts/skrl/play.py \
  --task=Baseline-UAVSwarm-Direct-v0 \
  --checkpoint <path.pt> --algorithm MAPPO --num_envs 32

# Evaluate individual agents and clone the best policy:
python scripts/skrl/eval_policy.py \
  --task=Baseline-UAVSwarm-Direct-v0 \
  --checkpoint <path.pt> --algorithm MAPPO
```

---

## Monitoring

```bash
# TensorBoard (run from IsaacLab root):
./isaaclab.sh -p -m tensorboard.main --logdir=logs
```

---

## Key Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_agents` | 5 | Crazyflie drones per environment |
| `scene.num_envs` | 256 | Parallel environments |
| `decimation` | 2 | Action repeats (effective 50 Hz) |
| `thrust_to_weight` | 1.9 | Thrust scaling |
| `moment_scale` | 0.01 | Moment/torque scaling |

MAPPO policy networks: `[128, 128]` hidden layers; critic: `[256, 256, 128]`. Target training: 5M timesteps.

---

## Repository Structure

```
source/UavSwarm/UavSwarm/tasks/direct/
├── baseline_uavswarm/       # Baseline env (no RM states)
│   ├── baseline_uavswarm_env.py
│   ├── baseline_uavswarm_env_cfg.py
│   └── agents/skrl_mappo_cfg.yaml
├── fulltask_swarm_rm/       # Full-task env (RM + all stages)
└── controllers/
    └── geometric_controller.py   # SE(3) geometric controller

scripts/
├── skrl/                    # skrl training / eval scripts
└── torchrl/                 # TorchRL training scripts
```

---

## License

- Isaac Lab components: BSD-3-Clause
- Package: Apache-2.0
