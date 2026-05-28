"""Formation geometry helpers for UAV swarm environments.

These functions operate on the environment's robot state and config
to compute swarm centroid and inverted-V formation positions.
"""

import torch


def compute_swarm_centroid(env) -> None:
    """Compute swarm centroid position for each environment (Stages 4 & 5).

    Calculates the mean position of all drones in each environment.
    Updates the env._swarm_centroid buffer: (num_envs, 3)
    """
    # Stack all drone positions: (num_envs, num_drones, 3)
    swarm_positions = torch.stack([rob.data.root_pos_w for rob in env._robots], dim=1)

    # Compute centroid as mean across all drones: (num_envs, 3)
    env._swarm_centroid = swarm_positions.mean(dim=1)


def get_inverted_v_formation(
    env,
    env_ids: torch.Tensor,
    env_origins: torch.Tensor,
    spawn_heights: torch.Tensor,
) -> torch.Tensor:
    """Calculate inverted V formation positions for all drones in specified environments.

    Args:
        env_ids: Indices of environments to reset
        env_origins: Origins of the environments being reset, shape (num_reset_envs, 3)
        spawn_heights: Height offset for each environment, shape (num_reset_envs,)

    Returns:
        Tensor of shape (num_reset_envs, num_drones, 3) with absolute world positions
        for each drone in each environment.
    """
    cfg = env.cfg
    num_drones = env.num_drones
    device = env.device

    num_reset_envs = len(env_ids)

    # Inverted V: apex at front (negative X), wings spread backward and outward
    v_angle_rad = torch.deg2rad(torch.tensor(cfg.swarm_cfg.formation_v_angle_deg, device=device))
    base_sep = cfg.swarm_cfg.formation_base_separation

    # Scale separation based on max_num_agents to ensure formation fits
    scale_factor = max(1.0, cfg.swarm_cfg.min_safe_distance / base_sep)
    effective_sep = base_sep * scale_factor

    # Generate formation positions template for each drone
    formation_template = torch.zeros(num_drones, 3, device=device)

    if num_drones == 1:
        # Single drone: centered at origin
        formation_template[0] = torch.tensor([0.0, 0.0, 0.0], device=device)
    else:
        # Multiple drones: inverted V formation
        # Apex drone (index 0) at the front (negative X)
        formation_template[0, 0] = cfg.swarm_cfg.formation_apex_offset
        formation_template[0, 1] = 0.0

        # Distribute remaining drones on left and right wings
        remaining_drones = num_drones - 1
        left_wing_count = remaining_drones // 2
        right_wing_count = remaining_drones - left_wing_count

        # Left wing (negative Y)
        for i in range(left_wing_count):
            wing_idx = i + 1
            x_offset = (i + 1) * effective_sep * torch.cos(v_angle_rad)
            y_offset = -(i + 1) * effective_sep * torch.sin(v_angle_rad)
            formation_template[wing_idx, 0] = cfg.swarm_cfg.formation_apex_offset + x_offset
            formation_template[wing_idx, 1] = y_offset

        # Right wing (positive Y)
        for i in range(right_wing_count):
            wing_idx = left_wing_count + i + 1
            x_offset = (i + 1) * effective_sep * torch.cos(v_angle_rad)
            y_offset = (i + 1) * effective_sep * torch.sin(v_angle_rad)
            formation_template[wing_idx, 0] = cfg.swarm_cfg.formation_apex_offset + x_offset
            formation_template[wing_idx, 1] = y_offset

    # Verify minimum separation constraint
    if num_drones > 1:
        dists = torch.cdist(formation_template.unsqueeze(0), formation_template.unsqueeze(0)).squeeze(0)
        dists = dists + torch.eye(num_drones, device=device) * 1000.0
        min_dist = dists.min()

        if min_dist < cfg.swarm_cfg.min_safe_distance:
            scale_up = cfg.swarm_cfg.min_safe_distance / min_dist
            formation_template[:, :2] *= scale_up

    # Expand template to all resetting environments
    formation_positions = formation_template.unsqueeze(0).expand(num_reset_envs, -1, -1).clone()

    # Add environment origins (XY) to all drones in each environment
    formation_positions[:, :, :2] += env_origins[:, :2].unsqueeze(1)

    # Add spawn heights (Z) to all drones in each environment
    formation_positions[:, :, 2] += spawn_heights.unsqueeze(1)

    return formation_positions  # Shape: (num_reset_envs, num_drones, 3)
