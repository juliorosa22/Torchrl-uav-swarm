"""Reward Machine state transitions for UAV swarm environments.

Computes per-agent RM states (H/S/C/O) based on altitude, obstacle proximity,
and neighbor distance. Uses cached sensor data populated by sensing.py.
"""

import torch


def switch_rm_state(env, all_positions: torch.Tensor) -> None:
    """Update Reward Machine states for all agents based on current conditions.

    State transitions:
    - Hovering (0): agent_z <= hover_min_altitude
    - Single-moving (1): agent_z > hover_min AND obstacle_dist > threshold AND neighbor_dist >= max_neighbor_distance
    - Coop-moving (2): agent_z > hover_min AND obstacle_dist > threshold AND neighbor_dist < max_neighbor_distance
    - Obstacle-avoiding (3): agent_z > hover_min AND obstacle_dist <= threshold

    Args:
        all_positions: Pre-computed robot positions (num_drones, num_envs, 3)

    Updates:
        env._rm_states: (num_envs, num_drones) tensor with state indices
    """
    if not env._cache_valid:
        raise RuntimeError(
            "switch_rm_state() called before cache populated! "
            "This should never happen in normal workflow."
        )

    cfg = env.cfg
    num_drones = env.num_drones
    num_envs = env.num_envs
    device = env.device

    # Extract altitudes: (num_drones, num_envs)
    all_z = all_positions[:, :, 2]

    # Use cached obstacle distances
    nearest_obstacle_dists = env._cached_obstacle_dists  # (num_drones, num_envs)

    # Compute neighbor distances from cached data
    if env.curriculum_stage in [1, 2, 3]:
        neighbor_dists = torch.full(
            (num_drones, num_envs),
            cfg.swarm_cfg.max_neighbor_distance,
            device=device,
        )
    else:
        # Stages 4-5: Extract distances from cached relative positions
        neighbor_dists = torch.linalg.norm(env._cached_neighbor_rel_pos_b, dim=2)

    # Initialize all states as Hovering (0)
    new_states = torch.zeros_like(all_z, dtype=torch.long)

    # Check conditions
    above_hover = all_z > cfg.reward_cfg.exit_hover_altitude
    far_from_obstacle = nearest_obstacle_dists > cfg.reward_cfg.enter_obstacle_avoidance_dist
    near_neighbor = neighbor_dists < cfg.reward_cfg.enter_coop_moving_dist

    # State 1 (S): Above hover AND far from obstacle AND far from neighbor
    single_moving_mask = above_hover & far_from_obstacle & (~near_neighbor)
    new_states[single_moving_mask] = 1

    # State 2 (C): Above hover AND far from obstacle AND near neighbor
    coop_moving_mask = above_hover & far_from_obstacle & near_neighbor
    new_states[coop_moving_mask] = 2

    # State 3 (O): Above hover AND close to obstacle (overrides states 1 & 2)
    obstacle_avoiding_mask = above_hover & (~far_from_obstacle)
    new_states[obstacle_avoiding_mask] = 3

    # State 0 (H): Below hover threshold (overrides all - highest priority)
    hovering_mask = ~above_hover
    new_states[hovering_mask] = 0

    # Update state buffer: (num_drones, num_envs) -> transpose -> (num_envs, num_drones)
    env._rm_states = new_states.transpose(0, 1)
