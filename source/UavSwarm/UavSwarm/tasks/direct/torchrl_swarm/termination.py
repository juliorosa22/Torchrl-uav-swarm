"""Termination conditions and goal checking for UAV swarm curriculum stages.

Handles collision, out-of-bounds, timeout, and stage-specific goal completion
checks. Also manages waypoint progression for stages 3 and 5.
"""

import torch


def get_dones(env) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Check termination conditions with curriculum-aware logic.

    Termination reasons:
    1. Collision: Any drone too low (< min_flight_height) or too high (> max_flight_height)
    2. Out of bounds: Any drone too far from environment origin
    3. Goal reached: All agents reached their goals (stage-dependent)
    4. Timeout: Episode exceeds max_episode_length

    Returns:
        Tuple of (terminated_dict, time_out_dict) where each is a dictionary
        mapping agent names to boolean tensors of shape (num_envs,)
    """
    # Timeout termination (all stages)
    time_out = env.episode_length_buf >= env.max_episode_length - 1

    # Collision termination (all stages)
    died_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for rob in env._robots:
        agent_z = rob.data.root_pos_w[:, 2]
        too_low = agent_z < env.cfg.reward_cfg.min_flight_height
        too_high = agent_z > env.cfg.reward_cfg.max_flight_height
        died_collision = died_collision | too_low | too_high

    # Out of bounds termination (all stages)
    died_out_of_bounds = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    env_origins = env.scene.env_origins
    max_distance_from_origin = env.cfg.reward_cfg.max_distance_from_origin

    for rob in env._robots:
        agent_pos_xy = rob.data.root_pos_w[:, :2]
        origin_xy = env_origins[:, :2]
        distance_from_origin = torch.linalg.norm(agent_pos_xy - origin_xy, dim=1)
        died_out_of_bounds = died_out_of_bounds | (distance_from_origin > max_distance_from_origin)

    # Goal reached termination (curriculum-aware)
    goal_reached = _check_goal_reached(env)

    # Combine termination conditions
    died = died_collision | died_out_of_bounds | goal_reached

    # Store termination reasons for logging in _reset_idx
    env._last_terminated = died
    env._last_timed_out = time_out

    if not hasattr(env, '_termination_reasons'):
        env._termination_reasons = {
            'collision': torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            'out_of_bounds': torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            'goal_reached': torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }

    env._termination_reasons['collision'] = died_collision
    env._termination_reasons['out_of_bounds'] = died_out_of_bounds
    env._termination_reasons['goal_reached'] = goal_reached

    terminated_dict = {f"robot_{i}": died for i in range(env.num_drones)}
    time_out_dict = {f"robot_{i}": time_out for i in range(env.num_drones)}

    return terminated_dict, time_out_dict


def _check_goal_reached(env) -> torch.Tensor:
    """Check if goals are reached based on current curriculum stage.

    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments
        have completed their goals.
    """
    stage = env.curriculum_stage

    if stage == 1:
        return _check_hover_goals_reached(env)
    elif stage == 2:
        return _check_individual_goals_reached(env)
    elif stage == 3:
        return _check_waypoint_goals_reached(env)
    elif stage == 4:
        return _check_swarm_goals_reached(env)
    elif stage == 5:
        return _check_swarm_waypoint_goals_reached(env)
    else:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _check_hover_goals_reached(env) -> torch.Tensor:
    """Check if all agents are hovering at their goal positions.

    Returns:
        Boolean tensor (num_envs,) - True if ALL agents within hover threshold
    """
    goal_reached_per_env = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for j, rob in enumerate(env._robots):
        distance_to_goal = torch.linalg.norm(
            env._desired_pos_w[:, j, :] - rob.data.root_pos_w,
            dim=1,
        )
        velocity_mag = torch.linalg.norm(rob.data.root_lin_vel_w, dim=1)

        agent_reached = (distance_to_goal < env.cfg.reward_cfg.hover_position_threshold) & \
                        (velocity_mag < env.cfg.reward_cfg.hover_velocity_threshold)

        goal_reached_per_env = goal_reached_per_env & agent_reached

    return goal_reached_per_env


def _check_individual_goals_reached(env) -> torch.Tensor:
    """Check if all agents reached their individual point-to-point goals.

    Returns:
        Boolean tensor (num_envs,) - True if ALL agents within goal threshold
    """
    goal_reached_per_env = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for j, rob in enumerate(env._robots):
        distance_to_goal = torch.linalg.norm(
            env._desired_pos_w[:, j, :] - rob.data.root_pos_w,
            dim=1,
        )
        agent_reached = distance_to_goal < env.cfg.reward_cfg.goal_position_threshold
        goal_reached_per_env = goal_reached_per_env & agent_reached

    return goal_reached_per_env


def _check_waypoint_goals_reached(env) -> torch.Tensor:
    """Check if all agents completed their waypoint paths (Stage 3).

    Returns:
        Boolean tensor (num_envs,) - True if ALL agents finished all waypoints
    """
    goal_reached_per_env = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for j in range(env.num_drones):
        completed = env._current_waypoint_idx[:, j] >= env.num_waypoints_per_agent
        goal_reached_per_env = goal_reached_per_env & completed

    return goal_reached_per_env


def _check_swarm_goals_reached(env) -> torch.Tensor:
    """Check if swarm reached goal formation (Stage 4).

    Returns:
        Boolean tensor (num_envs,) - True if swarm centroid within threshold
    """
    swarm_positions = torch.stack([rob.data.root_pos_w for rob in env._robots], dim=1)
    swarm_centroid = swarm_positions.mean(dim=1)
    goal_centroid = env._desired_pos_w.mean(dim=1)
    centroid_distance = torch.linalg.norm(swarm_centroid - goal_centroid, dim=1)

    return centroid_distance < env.cfg.reward_cfg.swarm_goal_threshold


def _check_swarm_waypoint_goals_reached(env) -> torch.Tensor:
    """Check if swarm completed waypoint path (Stage 5).

    Returns:
        Boolean tensor (num_envs,) - True if swarm finished all waypoints
    """
    return env._current_swarm_waypoint_idx >= env.num_swarm_waypoints


def update_waypoint_goals(env) -> None:
    """Vectorized waypoint update for all environments and agents (Stage 3).

    Checks distance from each agent to its current waypoint and advances
    to the next waypoint when within threshold.
    """
    all_positions = torch.stack([rob.data.root_pos_w for rob in env._robots], dim=0)
    all_positions = all_positions.transpose(0, 1)  # (num_envs, num_drones, 3)

    current_wp_idx = env._current_waypoint_idx  # (num_envs, num_drones)
    not_finished = current_wp_idx < env.num_waypoints_per_agent

    # Gather current waypoints
    env_idx = torch.arange(env.num_envs, device=env.device).view(-1, 1, 1)
    drone_idx = torch.arange(env.num_drones, device=env.device).view(1, -1, 1)
    wp_idx = current_wp_idx.unsqueeze(2).clamp(max=env.num_waypoints_per_agent - 1)
    coord_idx = torch.arange(3, device=env.device).view(1, 1, -1)

    current_waypoints = env._waypoint_paths[env_idx, drone_idx, wp_idx, coord_idx]
    distances = torch.linalg.norm(all_positions - current_waypoints, dim=2)

    reached = (distances < env.waypoint_reach_threshold) & not_finished

    env._current_waypoint_idx = torch.where(
        reached,
        torch.clamp(current_wp_idx + 1, max=env.num_waypoints_per_agent),
        current_wp_idx,
    )

    next_wp_idx = env._current_waypoint_idx.unsqueeze(2).clamp(max=env.num_waypoints_per_agent - 1)
    next_waypoints = env._waypoint_paths[env_idx, drone_idx, next_wp_idx, coord_idx]

    env._desired_pos_w = torch.where(
        reached.unsqueeze(2).expand(-1, -1, 3),
        next_waypoints,
        env._desired_pos_w,
    )


def update_swarm_waypoint_goals(env) -> None:
    """Update swarm goals based on centroid progress through waypoint path (Stage 5).

    When the swarm centroid reaches a waypoint, advances all agents to the next
    waypoint while maintaining their formation relative to the new target.
    """
    for env_idx in range(env.num_envs):
        current_wp_idx = env._current_swarm_waypoint_idx[env_idx].item()

        if current_wp_idx >= env.num_swarm_waypoints:
            continue

        swarm_positions = torch.stack(
            [rob.data.root_pos_w[env_idx] for rob in env._robots], dim=0
        )
        swarm_centroid = swarm_positions.mean(dim=0)

        current_waypoint = env._swarm_waypoint_paths[env_idx, current_wp_idx]
        distance_to_waypoint = torch.linalg.norm(swarm_centroid - current_waypoint).item()

        if distance_to_waypoint < env.swarm_waypoint_reach_threshold:
            next_wp_idx = current_wp_idx + 1

            if next_wp_idx < env.num_swarm_waypoints:
                env._current_swarm_waypoint_idx[env_idx] = next_wp_idx
                next_waypoint = env._swarm_waypoint_paths[env_idx, next_wp_idx]

                formation_offsets = swarm_positions - swarm_centroid.unsqueeze(0)

                for j in range(env.num_drones):
                    goal_pos = next_waypoint + formation_offsets[j]
                    env._desired_pos_w[env_idx, j, 0] = goal_pos[0]
                    env._desired_pos_w[env_idx, j, 1] = goal_pos[1]
                    env._desired_pos_w[env_idx, j, 2] = goal_pos[2]
            else:
                env._current_swarm_waypoint_idx[env_idx] = env.num_swarm_waypoints
