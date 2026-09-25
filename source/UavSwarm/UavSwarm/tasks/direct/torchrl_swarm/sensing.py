"""Sensing helpers for UAV swarm environments.

Provides cached nearest-neighbor, mean-neighbor, and nearest-obstacle computations.
All functions return body-frame tensors and are fully vectorised across drones and envs.
"""

import torch
from isaaclab.utils.math import quat_apply_inverse


def ensure_cache_populated(env) -> None:
    """Populate observation cache if invalid (lazy evaluation).

    Called by _get_rewards(), _get_observations(), and _get_states()
    to ensure cache is valid before use.
    """
    if env._cache_valid:
        return

    all_positions = torch.stack([rob.data.root_pos_w for rob in env._robots], dim=0)
    all_quats = torch.stack([rob.data.root_quat_w for rob in env._robots], dim=0)

    # Obstacle: distance (scalar) + bearing direction (unit vector, body frame)
    env._cached_obstacle_dists, env._cached_obstacle_dir_b = \
        _get_nearest_obstacle_vectorized(env, all_positions, all_quats)

    # Neighbour: nearest (pos + vel), mean-pooled (pos + vel), and (when
    # cfg.include_k_neighbors_in_obs) the K nearest individually -- all body frame.
    (env._cached_neighbor_rel_pos_b,
     env._cached_neighbor_rel_vel_b,
     env._cached_mean_neighbor_pos_b,
     env._cached_mean_neighbor_vel_b,
     env._cached_k_neighbor_pos_b,
     env._cached_k_neighbor_vel_b) = _get_neighbor_data_vectorized(env, all_positions, all_quats)

    env._cache_valid = True


def _get_neighbor_data_vectorized(
    env,
    all_positions: torch.Tensor,  # (num_drones, num_envs, 3)
    all_quats: torch.Tensor,       # (num_drones, num_envs, 4)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Vectorised nearest-neighbour and mean-neighbour computation.

    Returns all tensors in drone i's body frame so the policy never
    needs to handle world-frame coordinates.

    Returns:
        nearest_pos_b: (D, E, 3) — relative position of closest neighbour
        nearest_vel_b: (D, E, 3) — relative velocity of closest neighbour
        mean_pos_b:    (D, E, 3) — mean relative position of ALL other drones
                                   ≈ swarm-centroid offset in body frame
        mean_vel_b:    (D, E, 3) — mean relative velocity of ALL other drones
        k_nearest_pos_b: (D, E, K, 3) or None — the K nearest neighbours individually
                                   (nearest-first), only computed when
                                   cfg.include_k_neighbors_in_obs (stage 12,
                                   SwarmGravityAttn's attention policy). K =
                                   cfg.swarm_cfg.num_observed_neighbors, assumed < num_drones.
        k_nearest_vel_b: (D, E, K, 3) or None — matching relative velocities.
    """
    cfg = env.cfg
    num_drones = env.num_drones
    num_envs = env.num_envs
    device = env.device
    max_dist = cfg.swarm_cfg.max_neighbor_distance

    # Stages 1-3, 7: neighbour tracking inactive -- return sentinels. Also guard directly
    # on num_drones <= 1 regardless of stage (e.g. Formation/stage 6 run with
    # --num_agents 1): "mean of all OTHER drones" is undefined with zero other drones, and
    # the (num_drones - 1) division below produces NaN that splices straight into
    # _build_obs_tensor with no downstream guard, poisoning every observation from step 1.
    # Sentinel: neighbour is max_dist straight ahead (x-axis in world), zero relative vel.
    if env.curriculum_stage in [1, 2, 3, 7] or num_drones <= 1:
        default_w = torch.zeros(num_drones, num_envs, 3, device=device)
        default_w[:, :, 0] = max_dist
        default_b = quat_apply_inverse(
            all_quats.reshape(-1, 4), default_w.reshape(-1, 3)
        ).reshape(num_drones, num_envs, 3)
        zero_vel = torch.zeros_like(default_b)
        k_pos_b = k_vel_b = None
        if getattr(cfg, "include_k_neighbors_in_obs", False):
            k = cfg.swarm_cfg.num_observed_neighbors
            k_pos_b = default_b.unsqueeze(2).expand(-1, -1, k, -1).clone()
            k_vel_b = zero_vel.unsqueeze(2).expand(-1, -1, k, -1).clone()
        return default_b, zero_vel, default_b.clone(), zero_vel.clone(), k_pos_b, k_vel_b

    # diff[i, j, e, :] = pos_i - pos_j  →  shape (D, D, E, 3)
    diff = all_positions.unsqueeze(1) - all_positions.unsqueeze(0)

    # --- Nearest neighbour ---
    distances = torch.linalg.norm(diff, dim=3)  # (D, D, E)
    eye_mask = torch.eye(num_drones, device=device).unsqueeze(2)
    distances_masked = distances + eye_mask * 1e6
    nearest_idx = torch.argmin(distances_masked, dim=1)  # (D, E)

    env_idx = torch.arange(num_envs, device=device).unsqueeze(0).expand(num_drones, -1)

    # pos_j - pos_i = relative position toward nearest neighbour
    nearest_rel_pos_w = all_positions[nearest_idx, env_idx] - all_positions  # (D, E, 3)
    # Clamp magnitude to sensor range
    rn = nearest_rel_pos_w.norm(dim=2, keepdim=True)
    nearest_rel_pos_w = torch.where(
        rn > max_dist,
        nearest_rel_pos_w * (max_dist / (rn + 1e-8)),
        nearest_rel_pos_w,
    )

    all_lin_vels_w = torch.stack([rob.data.root_lin_vel_w for rob in env._robots], dim=0)
    nearest_rel_vel_w = all_lin_vels_w[nearest_idx, env_idx] - all_lin_vels_w  # (D, E, 3)
    # Clamp magnitude to sensor range -- same protection the position field above already
    # has (rn/max_dist clamp), now applied to velocity. Without this, a PhysX contact
    # impulse from an inter-agent collision can inject an unbounded velocity spike
    # straight into the observation the same step it happens (see
    # [[swarmgravity-rm-paper]] memory for how this was traced).
    max_vel = cfg.swarm_cfg.max_neighbor_velocity
    rvn = nearest_rel_vel_w.norm(dim=2, keepdim=True)
    nearest_rel_vel_w = torch.where(
        rvn > max_vel,
        nearest_rel_vel_w * (max_vel / (rvn + 1e-8)),
        nearest_rel_vel_w,
    )

    # --- Mean-pooled neighbours (swarm centroid signal) ---
    # -diff[i, j, e, :] = pos_j - pos_i; zero out self-pairs then mean over j
    eye_4d = torch.eye(num_drones, device=device).bool().unsqueeze(-1).unsqueeze(-1)  # (D, D, 1, 1)
    neg_diff = (-diff).masked_fill(eye_4d, 0.0)
    mean_rel_pos_w = neg_diff.sum(dim=1) / (num_drones - 1)  # (D, E, 3)

    # vel_j - vel_i for all pairs, zero self, mean over j
    # all_lin_vels_w.unsqueeze(0)[i,j,e,:] = vel_j  ;  unsqueeze(1)[i,j,e,:] = vel_i
    vel_diff = (all_lin_vels_w.unsqueeze(0) - all_lin_vels_w.unsqueeze(1))  # (D, D, E, 3)
    vel_diff_masked = vel_diff.masked_fill(eye_4d, 0.0)
    mean_rel_vel_w = vel_diff_masked.sum(dim=1) / (num_drones - 1)  # (D, E, 3)

    # Clamp mean position magnitude
    rm = mean_rel_pos_w.norm(dim=2, keepdim=True)
    mean_rel_pos_w = torch.where(
        rm > max_dist,
        mean_rel_pos_w * (max_dist / (rm + 1e-8)),
        mean_rel_pos_w,
    )

    # Clamp mean velocity magnitude -- same reasoning as nearest_rel_vel_w above; averaging
    # over neighbors dampens but does not eliminate a single collision-impulse outlier.
    rvm = mean_rel_vel_w.norm(dim=2, keepdim=True)
    mean_rel_vel_w = torch.where(
        rvm > max_vel,
        mean_rel_vel_w * (max_vel / (rvm + 1e-8)),
        mean_rel_vel_w,
    )

    # --- Transform all four tensors to body frame ---
    q_flat = all_quats.reshape(-1, 4)  # (D*E, 4)

    nearest_pos_b = quat_apply_inverse(
        q_flat, nearest_rel_pos_w.reshape(-1, 3)
    ).reshape(num_drones, num_envs, 3)

    nearest_vel_b = quat_apply_inverse(
        q_flat, nearest_rel_vel_w.reshape(-1, 3)
    ).reshape(num_drones, num_envs, 3)

    mean_pos_b = quat_apply_inverse(
        q_flat, mean_rel_pos_w.reshape(-1, 3)
    ).reshape(num_drones, num_envs, 3)

    mean_vel_b = quat_apply_inverse(
        q_flat, mean_rel_vel_w.reshape(-1, 3)
    ).reshape(num_drones, num_envs, 3)

    # --- K individual nearest neighbours (stage 12, SwarmGravityAttn's attention policy) ---
    # Reuses distances_masked/all_lin_vels_w already computed above for the nearest-neighbour
    # case. Only computed when needed -- cheap for tiny N, but no reason to pay it elsewhere.
    k_nearest_pos_b = k_nearest_vel_b = None
    if getattr(cfg, "include_k_neighbors_in_obs", False):
        k = cfg.swarm_cfg.num_observed_neighbors
        k_idx = torch.topk(distances_masked, k, dim=1, largest=False).indices  # (D, K, E)
        k_idx = k_idx.permute(0, 2, 1)  # (D, E, K)
        k_env_idx = env_idx.unsqueeze(-1).expand(-1, -1, k)  # (D, E, K)

        neighbor_pos_w = all_positions[k_idx, k_env_idx]  # (D, E, K, 3)
        k_rel_pos_w = neighbor_pos_w - all_positions.unsqueeze(2)
        rk = k_rel_pos_w.norm(dim=3, keepdim=True)
        k_rel_pos_w = torch.where(rk > max_dist, k_rel_pos_w * (max_dist / (rk + 1e-8)), k_rel_pos_w)

        neighbor_vel_w = all_lin_vels_w[k_idx, k_env_idx]  # (D, E, K, 3)
        k_rel_vel_w = neighbor_vel_w - all_lin_vels_w.unsqueeze(2)
        rvk = k_rel_vel_w.norm(dim=3, keepdim=True)
        k_rel_vel_w = torch.where(rvk > max_vel, k_rel_vel_w * (max_vel / (rvk + 1e-8)), k_rel_vel_w)

        q_k = all_quats.unsqueeze(2).expand(-1, -1, k, -1).reshape(-1, 4)
        k_nearest_pos_b = quat_apply_inverse(q_k, k_rel_pos_w.reshape(-1, 3)).reshape(num_drones, num_envs, k, 3)
        k_nearest_vel_b = quat_apply_inverse(q_k, k_rel_vel_w.reshape(-1, 3)).reshape(num_drones, num_envs, k, 3)

    return nearest_pos_b, nearest_vel_b, mean_pos_b, mean_vel_b, k_nearest_pos_b, k_nearest_vel_b


def _get_nearest_obstacle_vectorized(
    env,
    all_positions: torch.Tensor,  # (num_drones, num_envs, 3)
    all_quats: torch.Tensor,       # (num_drones, num_envs, 4)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised obstacle sensing: scalar distance + body-frame bearing.

    Returns:
        min_distances: (D, E)    — distance to nearest obstacle, clamped to max range
        obs_dir_b:     (D, E, 3) — unit vector pointing toward nearest obstacle in body frame.
                                   Defaults to (1, 0, 0) body-forward when no obstacles present.
    """
    num_drones = env.num_drones
    num_envs = env.num_envs
    device = env.device
    max_dist = env.cfg.curriculum.max_obstacle_distance

    # No obstacles — return sentinel: far away, direction straight ahead (body +x)
    if env._obstacle_positions is None or env.curriculum_stage not in [3, 5]:
        min_distances = torch.full((num_drones, num_envs), max_dist, device=device)
        obs_dir_b = torch.zeros(num_drones, num_envs, 3, device=device)
        obs_dir_b[:, :, 0] = 1.0  # +x body-forward sentinel
        return min_distances, obs_dir_b

    # agent_pos: (D, E, 1, 3),  obs_pos: (1, 1, N_obs, 3)
    agent_expanded = all_positions.unsqueeze(2)
    obs_expanded = env._obstacle_positions.unsqueeze(0).unsqueeze(0)

    # diff[d, e, k, :] = agent_pos[d,e] - obs_pos[k]  →  shape (D, E, N_obs, 3)
    diff = agent_expanded - obs_expanded
    distances = diff.norm(dim=3)  # (D, E, N_obs)

    # Nearest obstacle index and distance
    min_dist_values, nearest_obs_idx = distances.min(dim=2)  # (D, E) each
    min_distances = min_dist_values.clamp(0.0, max_dist)

    # Direction toward nearest obstacle: obs_pos - agent_pos = -diff at nearest_obs_idx
    d_idx = torch.arange(num_drones, device=device).view(-1, 1).expand(-1, num_envs)
    e_idx = torch.arange(num_envs, device=device).view(1, -1).expand(num_drones, -1)
    nearest_diff = diff[d_idx, e_idx, nearest_obs_idx, :]  # (D, E, 3) = agent - obs
    obs_dir_w = -nearest_diff  # toward obstacle
    obs_dir_norm_w = obs_dir_w / (obs_dir_w.norm(dim=2, keepdim=True) + 1e-8)

    obs_dir_b = quat_apply_inverse(
        all_quats.reshape(-1, 4),
        obs_dir_norm_w.reshape(-1, 3),
    ).reshape(num_drones, num_envs, 3)

    return min_distances, obs_dir_b
