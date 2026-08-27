"""Energy-based reward function for UAV swarm curriculum.

Combines position energy, distance progress delta, velocity alignment,
smoothness coupling, obstacle repulsion, and cooperation potential.
All components are weighted by RM state for context-aware shaping.

Constants are defined at module level for clarity. These should be
moved to RewardMachineCfg in a future config restructuring pass.
"""

import torch

from .formation import compute_swarm_centroid

# Position energy
K_POS = 20.0

# Distance delta (progress indicator)
K_DELTA = 3.0

# Velocity alignment (direction efficiency)
K_ALIGN = 2.0

# Smoothness term (velocity coupling penalty)
ALPHA = 0.3
BETA = 0.5
GAMMA = 0.8

# Obstacle term (continuous repulsive potential)
D_SAFE = 1.5
D_INFLUENCE = 3.0
K_OBS = 3.0

# Cooperation term (Laplace potential)
D_OPT = 1.75
K_COOP = 2.0

# RM state-aware weights
W_HOVER = (1.2, 0.3, 0.0, 1.5)
W_SINGLE = (0.8, 1.5, 1.2, 0.7)
W_COOP = (0.6, 1.0, 0.8, 1.0)
W_AVOID = (0.5, 0.5, 0.3, 1.2)

# Output scaling
REWARD_SCALE = 0.15

# Safety penalties
COLLISION_PENALTY = 10.0
JERK_PENALTY_SCALE = 0.01
LIN_VEL_PENALTY_SCALE = 0.005
ANG_VEL_PENALTY_SCALE = 0.0025

# Formation-assignment task (stage 6): inter-agent safety penalty, distinct from the
# K_COOP travel potential above (that one rewards staying near an in-transit neighbor;
# this one only penalizes violating min_safe_distance while converging to an assigned slot).
K_FORM_SAFETY = 5.0


def get_rewards(env) -> dict[str, torch.Tensor]:
    """Energy-based reward with distance delta, velocity alignment, and RM state shaping.

    Returns:
        Dictionary mapping agent names to reward tensors of shape (num_envs,).
    """
    from .sensing import ensure_cache_populated

    ensure_cache_populated(env)

    # Stack all robot data: (num_drones, num_envs, 3)
    all_positions = torch.stack([rob.data.root_pos_w for rob in env._robots], dim=0)
    all_lin_vels = torch.stack([rob.data.root_lin_vel_b for rob in env._robots], dim=0)
    all_ang_vels = torch.stack([rob.data.root_ang_vel_b for rob in env._robots], dim=0)
    all_lin_vels_w = torch.stack([rob.data.root_lin_vel_w for rob in env._robots], dim=0)

    desired_transposed = env._desired_pos_w.transpose(0, 1)  # (num_drones, num_envs, 3)

    # 1. POSITION ENERGY (Inverse Quadratic Potential)
    distances = torch.linalg.norm(desired_transposed - all_positions, dim=2)
    position_energy = K_POS / (1.0 + distances ** 2)

    # 2. DISTANCE DELTA (Progress Indicator)
    if not hasattr(env, '_prev_distances'):
        env._prev_distances = distances.clone()

    distance_delta = env._prev_distances - distances
    env._prev_distances = distances.clone()
    delta_term = K_DELTA * distance_delta

    # 3. VELOCITY ALIGNMENT (Direction Efficiency)
    goal_directions = desired_transposed - all_positions
    goal_dist = torch.linalg.norm(goal_directions, dim=2, keepdim=True) + 1e-8
    goal_directions_norm = goal_directions / goal_dist

    vel_mag = torch.linalg.norm(all_lin_vels_w, dim=2, keepdim=True) + 1e-8
    vel_directions_norm = all_lin_vels_w / vel_mag

    cos_alignment = torch.sum(goal_directions_norm * vel_directions_norm, dim=2)
    is_moving = (vel_mag.squeeze(-1) > 0.1).float()
    alignment_term = K_ALIGN * torch.clamp(cos_alignment, min=0.0) * is_moving

    # 4. SMOOTHNESS TERM (Velocity Coupling Penalty)
    lin_vel_mag = torch.linalg.norm(all_lin_vels, dim=2)
    ang_vel_mag = torch.linalg.norm(all_ang_vels, dim=2)

    smoothness_multiplier = 1.0 / (
        1.0 +
        ALPHA * lin_vel_mag +
        BETA * ang_vel_mag +
        GAMMA * lin_vel_mag * ang_vel_mag
    )

    # 5. OBSTACLE TERM (Continuous Repulsive Potential)
    if env.curriculum_stage in [3, 5]:
        obstacle_dists = env._cached_obstacle_dists

        influence = torch.clamp(
            (D_INFLUENCE - obstacle_dists) / (D_INFLUENCE - D_SAFE),
            0.0,
            1.0,
        )

        obstacle_penalty = torch.exp(-K_OBS * influence ** 2)
    else:
        obstacle_penalty = torch.ones_like(position_energy)

    # 6. COOPERATION TERM (Laplace Potential)
    if env.curriculum_stage in [4, 5]:
        diff = all_positions.unsqueeze(1) - all_positions.unsqueeze(0)
        pairwise_dists = torch.linalg.norm(diff, dim=3)

        eye_mask = torch.eye(env.num_drones, device=env.device).unsqueeze(2)
        pairwise_dists = pairwise_dists + eye_mask * 1e6

        neighbor_dists = pairwise_dists.min(dim=1)[0]

        deviation = (neighbor_dists - D_OPT) ** 2
        coop_penalty = torch.exp(-K_COOP * deviation)
    else:
        coop_penalty = torch.ones_like(position_energy)

    # 7. RM STATE-AWARE WEIGHTING
    rm_states = env._rm_states.transpose(0, 1)  # (num_drones, num_envs)

    w_position = torch.ones_like(position_energy)
    w_delta = torch.ones_like(position_energy)
    w_alignment = torch.ones_like(position_energy)
    w_smoothness = torch.ones_like(position_energy)

    # RM STATE 0: HOVERING
    is_hovering = (rm_states == 0).float()
    w_position = torch.where(is_hovering.bool(), torch.full_like(w_position, W_HOVER[0]), w_position)
    w_delta = torch.where(is_hovering.bool(), torch.full_like(w_delta, W_HOVER[1]), w_delta)
    w_alignment = torch.where(is_hovering.bool(), torch.full_like(w_alignment, W_HOVER[2]), w_alignment)
    w_smoothness = torch.where(is_hovering.bool(), torch.full_like(w_smoothness, W_HOVER[3]), w_smoothness)

    # RM STATE 1: SINGLE-MOVING
    is_single = (rm_states == 1).float()
    w_position = torch.where(is_single.bool(), torch.full_like(w_position, W_SINGLE[0]), w_position)
    w_delta = torch.where(is_single.bool(), torch.full_like(w_delta, W_SINGLE[1]), w_delta)
    w_alignment = torch.where(is_single.bool(), torch.full_like(w_alignment, W_SINGLE[2]), w_alignment)
    w_smoothness = torch.where(is_single.bool(), torch.full_like(w_smoothness, W_SINGLE[3]), w_smoothness)

    # RM STATE 2: COOP-MOVING
    is_coop = (rm_states == 2).float()
    w_position = torch.where(is_coop.bool(), torch.full_like(w_position, W_COOP[0]), w_position)
    w_delta = torch.where(is_coop.bool(), torch.full_like(w_delta, W_COOP[1]), w_delta)
    w_alignment = torch.where(is_coop.bool(), torch.full_like(w_alignment, W_COOP[2]), w_alignment)
    w_smoothness = torch.where(is_coop.bool(), torch.full_like(w_smoothness, W_COOP[3]), w_smoothness)

    # RM STATE 3: OBSTACLE-AVOIDING
    is_avoiding = (rm_states == 3).float()
    w_position = torch.where(is_avoiding.bool(), torch.full_like(w_position, W_AVOID[0]), w_position)
    w_delta = torch.where(is_avoiding.bool(), torch.full_like(w_delta, W_AVOID[1]), w_delta)
    w_alignment = torch.where(is_avoiding.bool(), torch.full_like(w_alignment, W_AVOID[2]), w_alignment)
    w_smoothness = torch.where(is_avoiding.bool(), torch.full_like(w_smoothness, W_AVOID[3]), w_smoothness)

    # 8. COMBINED ENERGY REWARD (Hybrid Additive + Multiplicative)
    base_energy = (
        w_position * position_energy +
        w_delta * delta_term +
        w_alignment * alignment_term
    )

    combined_energy = (
        base_energy *
        (w_smoothness * smoothness_multiplier) *
        obstacle_penalty *
        coop_penalty
    )

    bounded_reward = 2.0 * torch.tanh(REWARD_SCALE * combined_energy)

    # 9. AGGREGATE + SAFETY PENALTIES
    mean_reward_per_env = bounded_reward.mean(dim=0)

    # Collision penalty
    agent_z = all_positions[:, :, 2]
    too_low = agent_z < env.cfg.reward_cfg.min_flight_height
    too_high = agent_z > env.cfg.reward_cfg.max_flight_height
    collision = -(too_low | too_high).any(dim=0).float() * COLLISION_PENALTY

    # Jerk penalty
    if not hasattr(env, '_prev_actions'):
        env._prev_actions = torch.zeros_like(env._actions)

    action_diff = env._actions - env._prev_actions
    jerk_penalty = torch.sum(action_diff ** 2, dim=(1, 2)) * -JERK_PENALTY_SCALE

    env._prev_actions = env._actions.clone()

    # Small velocity penalties
    lin_vel_penalty = lin_vel_mag.mean(dim=0) * -LIN_VEL_PENALTY_SCALE
    ang_vel_penalty = ang_vel_mag.mean(dim=0) * -ANG_VEL_PENALTY_SCALE

    reward = mean_reward_per_env + collision + jerk_penalty + lin_vel_penalty + ang_vel_penalty

    # Guard against NaN/Inf that can occur when physics diverges (drone tumbling after collision).
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-COLLISION_PENALTY)

    # 10. LOGGING
    # Diagnostic decomposition: compute what each component contributes.
    # The actual reward is multiplicative, so we approximate additive components
    # by computing the reward with each penalty term isolated.
    dist_reward = 2.0 * torch.tanh(REWARD_SCALE * base_energy * (w_smoothness * smoothness_multiplier))
    obs_component = dist_reward * (1.0 - obstacle_penalty)
    coop_component = dist_reward * (1.0 - coop_penalty)

    # Formation error: mean distance from swarm centroid (stages 4-5)
    if env.curriculum_stage in [4, 5]:
        compute_swarm_centroid(env)
        formation_error = torch.linalg.norm(
            all_positions - env._swarm_centroid.unsqueeze(0), dim=2
        ).mean(dim=0)
    else:
        formation_error = torch.zeros_like(reward)

    env._metrics.update(
        distance_to_goal=distances.mean(dim=0),
        lin_vel=lin_vel_penalty.abs(),
        ang_vel=ang_vel_penalty.abs(),
        collision=collision.abs(),
        mean_reward=reward,
        dist_component=dist_reward.mean(dim=0),
        obs_component=obs_component.mean(dim=0),
        coop_component=coop_component.mean(dim=0),
        formation=formation_error,
        swarm_cohesion=formation_error,  # cohesion = same metric as formation for now
    )

    return {f"robot_{i}": reward for i in range(env.num_drones)}


def get_formation_rewards(env) -> dict[str, torch.Tensor]:
    """Purely additive formation-assignment reward (curriculum stage 6).

    Reach and hold the Hungarian-assigned V-formation slot. No RM-state weighting --
    this is the plain MARL baseline arm; the MARL+RM comparison arm for this task is a
    follow-up. Mirrors the additive-reward design used for the baseline navigation task.
    """
    from .sensing import ensure_cache_populated

    ensure_cache_populated(env)

    all_positions = torch.stack([rob.data.root_pos_w for rob in env._robots], dim=0)
    all_lin_vels_w = torch.stack([rob.data.root_lin_vel_w for rob in env._robots], dim=0)

    desired_transposed = env._desired_pos_w.transpose(0, 1)  # (num_drones, num_envs, 3)

    # 1. POSITION ENERGY toward assigned slot
    distances = torch.linalg.norm(desired_transposed - all_positions, dim=2)
    position_energy = K_POS / (1.0 + distances ** 2)

    # 2. DISTANCE DELTA (progress indicator)
    if not hasattr(env, '_prev_distances'):
        env._prev_distances = distances.clone()

    distance_delta = env._prev_distances - distances
    env._prev_distances = distances.clone()
    delta_term = K_DELTA * distance_delta

    # 3. VELOCITY ALIGNMENT toward assigned slot
    goal_directions = desired_transposed - all_positions
    goal_dist = torch.linalg.norm(goal_directions, dim=2, keepdim=True) + 1e-8
    goal_directions_norm = goal_directions / goal_dist

    vel_mag = torch.linalg.norm(all_lin_vels_w, dim=2, keepdim=True) + 1e-8
    vel_directions_norm = all_lin_vels_w / vel_mag

    cos_alignment = torch.sum(goal_directions_norm * vel_directions_norm, dim=2)
    is_moving = (vel_mag.squeeze(-1) > 0.1).float()
    alignment_term = K_ALIGN * torch.clamp(cos_alignment, min=0.0) * is_moving

    # 4. INTER-AGENT SAFETY PENALTY (nearest-neighbor threshold, not the travel potential)
    neighbor_dists = torch.linalg.norm(env._cached_neighbor_rel_pos_b, dim=2)  # (num_drones, num_envs)
    min_safe = env.cfg.swarm_cfg.min_safe_distance
    safety_violation = torch.clamp(min_safe - neighbor_dists, min=0.0)
    safety_penalty = -K_FORM_SAFETY * safety_violation ** 2

    per_drone_reward = position_energy + delta_term + alignment_term + safety_penalty
    mean_reward_per_env = per_drone_reward.mean(dim=0)

    # 5. HARD COLLISION PENALTY (ground / ceiling)
    agent_z = all_positions[:, :, 2]
    too_low = agent_z < env.cfg.reward_cfg.min_flight_height
    too_high = agent_z > env.cfg.reward_cfg.max_flight_height
    collision = -(too_low | too_high).any(dim=0).float() * COLLISION_PENALTY

    # 6. JERK PENALTY (action smoothness)
    if not hasattr(env, '_prev_actions'):
        env._prev_actions = torch.zeros_like(env._actions)

    action_diff = env._actions - env._prev_actions
    jerk_penalty = torch.sum(action_diff ** 2, dim=(1, 2)) * -JERK_PENALTY_SCALE
    env._prev_actions = env._actions.clone()

    reward = mean_reward_per_env + collision + jerk_penalty
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-COLLISION_PENALTY)

    # 7. LOGGING -- formation error = mean per-agent distance to assigned slot
    formation_error = distances.mean(dim=0)

    env._metrics.update(
        distance_to_goal=distances.mean(dim=0),
        collision=collision.abs(),
        mean_reward=reward,
        dist_component=mean_reward_per_env,
        formation=formation_error,
        swarm_cohesion=formation_error,
    )

    return {f"robot_{i}": reward for i in range(env.num_drones)}
