"""Sensing helpers for UAV swarm environments.

Provides cached nearest-neighbor and nearest-obstacle distance computations,
plus Reward Machine state transitions based on those cached values.
"""

import torch
from isaaclab.utils.math import quat_apply_inverse


def ensure_cache_populated(env) -> None:
    """Populate cache if invalid (lazy evaluation).

    Called by _get_rewards(), _get_observations(), and _get_states()
    to ensure cache is valid before use.
    """
    if env._cache_valid:
        return

    # Stack all robot data: (num_drones, num_envs, 3)
    all_positions = torch.stack([rob.data.root_pos_w for rob in env._robots], dim=0)
    all_quats = torch.stack([rob.data.root_quat_w for rob in env._robots], dim=0)

    # Obstacle distances: (num_drones, num_envs)
    env._cached_obstacle_dists = _get_nearest_obstacle_distance_vectorized(env, all_positions)

    # Neighbor data: (num_drones, num_envs, 3)
    env._cached_neighbor_rel_pos_b, env._cached_neighbor_rel_vel_b = \
        _get_nearest_neighbor_data_vectorized(env, all_positions, all_quats)

    # Mark cache as valid
    env._cache_valid = True


def _get_nearest_neighbor_data_vectorized(
    env,
    all_positions: torch.Tensor,  # (num_drones, num_envs, 3)
    all_quats: torch.Tensor,       # (num_drones, num_envs, 4)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fully vectorized nearest neighbor calculation.

    Returns:
        relative_pos_b: (num_drones, num_envs, 3) - relative position in body frame
        relative_vel_b: (num_drones, num_envs, 3) - relative velocity in body frame
    """
    cfg = env.cfg
    num_drones = env.num_drones
    num_envs = env.num_envs
    device = env.device

    if env.curriculum_stage in [1, 2, 3]:
        default_rel_pos_w = torch.zeros(num_drones, num_envs, 3, device=device)
        default_rel_pos_w[:, :, 0] = cfg.swarm_cfg.max_neighbor_distance

        default_rel_pos_b = quat_apply_inverse(
            all_quats.reshape(-1, 4),
            default_rel_pos_w.reshape(-1, 3)
        ).reshape(num_drones, num_envs, 3)

        default_rel_vel_b = torch.zeros_like(default_rel_pos_b)
        return default_rel_pos_b, default_rel_vel_b

    # Compute pairwise distances
    diff = all_positions.unsqueeze(1) - all_positions.unsqueeze(0)  # (D, D, E, 3)
    distances = torch.linalg.norm(diff, dim=3)  # (D, D, E)

    # Mask self-distances
    eye_mask = torch.eye(num_drones, device=device).unsqueeze(2)
    distances = distances + eye_mask * 1e6

    # Find nearest neighbors: (D, E)
    nearest_idx = torch.argmin(distances, dim=1)

    # Create environment indices: (D, E)
    env_idx = torch.arange(num_envs, device=device).unsqueeze(0).expand(num_drones, -1)

    # Gather positions: all_positions[nearest_idx[i,j], env_idx[i,j], :]
    nearest_pos = all_positions[nearest_idx, env_idx, :]  # (D, E, 3)

    # Relative position
    relative_pos_w = nearest_pos - all_positions

    # Clamp magnitude
    rel_pos_norm = torch.linalg.norm(relative_pos_w, dim=2, keepdim=True)
    relative_pos_w = torch.where(
        rel_pos_norm > cfg.swarm_cfg.max_neighbor_distance,
        relative_pos_w * (cfg.swarm_cfg.max_neighbor_distance / (rel_pos_norm + 1e-8)),
        relative_pos_w,
    )

    # Transform to body frame
    relative_pos_b = quat_apply_inverse(
        all_quats.reshape(-1, 4),
        relative_pos_w.reshape(-1, 3)
    ).reshape(num_drones, num_envs, 3)

    # Same for velocity
    all_lin_vels = torch.stack([rob.data.root_lin_vel_w for rob in env._robots], dim=0)
    nearest_vel = all_lin_vels[nearest_idx, env_idx, :]

    relative_vel_w = nearest_vel - all_lin_vels
    relative_vel_b = quat_apply_inverse(
        all_quats.reshape(-1, 4),
        relative_vel_w.reshape(-1, 3)
    ).reshape(num_drones, num_envs, 3)

    return relative_pos_b, relative_vel_b


def _get_nearest_obstacle_distance_vectorized(
    env,
    all_positions: torch.Tensor,  # (num_drones, num_envs, 3)
) -> torch.Tensor:
    """Vectorized obstacle distance calculation for all agents.

    Args:
        all_positions: Agent positions, shape (num_drones, num_envs, 3)

    Returns:
        Nearest obstacle distances, shape (num_drones, num_envs)
        Clamped to [0, max_obstacle_distance]
    """
    if env._obstacle_positions is None or env.curriculum_stage not in [3, 5]:
        return torch.full(
            (env.num_drones, env.num_envs),
            env.cfg.curriculum.max_obstacle_distance,
            device=env.device,
        )

    # Expand dimensions for broadcasting
    # agent_pos: (num_drones, num_envs, 1, 3)
    # obs_pos:   (1, 1, num_obstacles, 3)
    agent_pos_expanded = all_positions.unsqueeze(2)
    obs_pos_expanded = env._obstacle_positions.unsqueeze(0).unsqueeze(0)

    # Calculate distances: (num_drones, num_envs, num_obstacles)
    diff = agent_pos_expanded - obs_pos_expanded
    distances = torch.linalg.norm(diff, dim=3)

    # Find minimum distance to any obstacle: (num_drones, num_envs)
    min_distances = distances.min(dim=2)[0]

    # Clamp to maximum range
    min_distances = torch.clamp(min_distances, 0.0, env.cfg.curriculum.max_obstacle_distance)

    return min_distances
