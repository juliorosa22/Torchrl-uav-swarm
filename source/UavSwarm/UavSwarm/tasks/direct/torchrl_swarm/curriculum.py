"""Curriculum stage position setters for UAV swarm environments.

Each stage has a dedicated function that resets agent start positions and
goal positions according to the curriculum progression:
  Stage 1: Hover (vertical stability)
  Stage 2: Point-to-point navigation
  Stage 3: Obstacle course (per-agent waypoints)
  Stage 4: Swarm formation navigation
  Stage 5: Swarm + obstacles (formation through stacked-X pattern)
  Stage 6: Formation-assignment scalability task (scatter-spawn -> Hungarian-assigned
           V-formation slots). Internal dispatch value only; not part of the sequential
           1-5 curriculum, used by the standalone Formation-TorchRL-UAVSwarm-Direct-v0 task.
  Stage 7: Single-goal navigation (independent random goal per agent, no assignment).
           Internal dispatch value only, used by SingleGoal-TorchRL-UAVSwarm-Direct-v0.
  Stage 8: Swarm-gravity (one shared target per env, no fixed formation shape -- agents
           self-organize around it). Internal dispatch value only, used by
           SwarmGravity-TorchRL-UAVSwarm-Direct-v0.
"""

import torch


def set_stage1_positions(env, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
    """Set hover goals - grid-based start positions with goals directly above.

    Spawns drones at stage1_spawn_height_range (1.5–2.5 m by default) so that
    a random descending policy has several seconds of margin before hitting the
    min_flight_height termination floor (0.1 m). Goal height is spawn + a random
    delta from stage1_goal_height_delta_range, keeping task difficulty consistent
    regardless of where the drone spawns.
    """
    num_reset_envs = len(env_ids)
    cfg_c = env.cfg.curriculum

    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(env.num_drones, dtype=torch.float32))))
    spacing = torch.zeros(1, device=env.device).uniform_(
        cfg_c.spawn_grid_spacing_range[0],
        cfg_c.spawn_grid_spacing_range[1],
    )

    spawn_lo, spawn_hi = cfg_c.stage1_spawn_height_range
    delta_lo, delta_hi = cfg_c.stage1_goal_height_delta_range

    for env_idx in range(num_reset_envs):
        perm = torch.randperm(env.num_drones, device=env.device)

        start_heights = torch.zeros(env.num_drones, device=env.device).uniform_(spawn_lo, spawn_hi)
        goal_deltas = torch.zeros(env.num_drones, device=env.device).uniform_(delta_lo, delta_hi)

        for j, rob in enumerate(env._robots):
            env_id_single = env_ids[env_idx].unsqueeze(0)

            grid_idx = perm[j].item()
            grid_x = (grid_idx % grid_size) * spacing - (grid_size * spacing / 2.0)
            grid_y = (grid_idx // grid_size) * spacing - (grid_size * spacing / 2.0)

            joint_pos = rob.data.default_joint_pos[env_id_single]
            joint_vel = rob.data.default_joint_vel[env_id_single]
            default_root_state = rob.data.default_root_state[env_id_single].clone()

            default_root_state[:, 0] = env_origins[env_idx, 0] + grid_x
            default_root_state[:, 1] = env_origins[env_idx, 1] + grid_y
            default_root_state[:, 2] = start_heights[j]

            rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
            rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
            rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)

            xy_noise = torch.zeros(2, device=env.device).uniform_(-0.05, 0.05)
            env._desired_pos_w[env_id_single, j, 0] = default_root_state[0, 0] + xy_noise[0]
            env._desired_pos_w[env_id_single, j, 1] = default_root_state[0, 1] + xy_noise[1]
            env._desired_pos_w[env_id_single, j, 2] = start_heights[j] + goal_deltas[j]


def set_stage2_positions(env, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
    """Set individual point-to-point goals for curriculum stage 2.

    Agents must learn to:
    1. Navigate stably in XY plane at different heights
    2. Rotate (yaw) to face the goal direction
    3. Reach distant goals while avoiding inter-agent collisions
    """
    num_reset_envs = len(env_ids)

    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(env.num_drones, dtype=torch.float32))))
    spacing = torch.zeros(1, device=env.device).uniform_(
        env.cfg.curriculum.spawn_grid_spacing_range[0],
        env.cfg.curriculum.spawn_grid_spacing_range[1],
    )

    min_height = env.cfg.curriculum.goal_height_range[0]
    max_height = env.cfg.curriculum.goal_height_range[1]

    z_spacing = env.cfg.curriculum.stage2_zdist_xy_plane
    spawn_lo, spawn_hi = env.cfg.curriculum.stage2_spawn_height_range
    base_height = torch.zeros(1, device=env.device).uniform_(spawn_lo, spawn_hi).item()

    for env_idx in range(num_reset_envs):
        perm = torch.randperm(env.num_drones, device=env.device)

        heights = torch.arange(env.num_drones, device=env.device, dtype=torch.float32)
        heights = base_height + heights * z_spacing
        heights = torch.clamp(heights, min=min_height, max=max_height)

        height_perm = torch.randperm(env.num_drones, device=env.device)
        assigned_heights = heights[height_perm]

        for j, rob in enumerate(env._robots):
            env_id_single = env_ids[env_idx].unsqueeze(0)

            grid_idx = perm[j].item()
            grid_x = (grid_idx % grid_size) * spacing - (grid_size * spacing / 2.0)
            grid_y = (grid_idx // grid_size) * spacing - (grid_size * spacing / 2.0)

            joint_pos = rob.data.default_joint_pos[env_id_single]
            joint_vel = rob.data.default_joint_vel[env_id_single]
            default_root_state = rob.data.default_root_state[env_id_single].clone()

            start_x = env_origins[env_idx, 0] + grid_x
            start_y = env_origins[env_idx, 1] + grid_y
            start_z = assigned_heights[j].item()

            default_root_state[:, 0] = start_x
            default_root_state[:, 1] = start_y
            default_root_state[:, 2] = start_z

            rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
            rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
            rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)

            goal_distance = torch.zeros(1, device=env.device).uniform_(
                2.0,
                env.cfg.curriculum.stage2_goal_distance,
            ).item()

            goal_angle = torch.zeros(1, device=env.device).uniform_(
                0.0,
                2.0 * torch.pi,
            ).item()

            goal_offset_x = goal_distance * torch.cos(torch.tensor(goal_angle, device=env.device))
            goal_offset_y = goal_distance * torch.sin(torch.tensor(goal_angle, device=env.device))

            goal_x = start_x + goal_offset_x
            goal_y = start_y + goal_offset_y

            goal_z_noise = torch.zeros(1, device=env.device).uniform_(-0.3, 0.3).item()
            goal_z = torch.clamp(
                torch.tensor(start_z + goal_z_noise, device=env.device),
                min=min_height,
                max=max_height,
            ).item()

            env._desired_pos_w[env_id_single, j, 0] = goal_x
            env._desired_pos_w[env_id_single, j, 1] = goal_y
            env._desired_pos_w[env_id_single, j, 2] = goal_z


def set_singlegoal_positions(env, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
    """Set independent point-to-point goals for curriculum stage 7 (single-UAV diagnostic).

    Same scatter-spawn + independent-random-goal pattern as set_stage2_positions, using
    stage7_* config fields -- no Hungarian/V-formation assignment, unlike stage 6.
    """
    num_reset_envs = len(env_ids)

    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(env.num_drones, dtype=torch.float32))))
    spacing = torch.zeros(1, device=env.device).uniform_(
        env.cfg.curriculum.spawn_grid_spacing_range[0],
        env.cfg.curriculum.spawn_grid_spacing_range[1],
    )

    min_height = env.cfg.curriculum.goal_height_range[0]
    max_height = env.cfg.curriculum.goal_height_range[1]

    z_spacing = env.cfg.curriculum.stage7_zdist_xy_plane
    spawn_lo, spawn_hi = env.cfg.curriculum.stage7_spawn_height_range
    base_height = torch.zeros(1, device=env.device).uniform_(spawn_lo, spawn_hi).item()

    for env_idx in range(num_reset_envs):
        perm = torch.randperm(env.num_drones, device=env.device)

        heights = torch.arange(env.num_drones, device=env.device, dtype=torch.float32)
        heights = base_height + heights * z_spacing
        heights = torch.clamp(heights, min=min_height, max=max_height)

        height_perm = torch.randperm(env.num_drones, device=env.device)
        assigned_heights = heights[height_perm]

        for j, rob in enumerate(env._robots):
            env_id_single = env_ids[env_idx].unsqueeze(0)

            grid_idx = perm[j].item()
            grid_x = (grid_idx % grid_size) * spacing - (grid_size * spacing / 2.0)
            grid_y = (grid_idx // grid_size) * spacing - (grid_size * spacing / 2.0)

            joint_pos = rob.data.default_joint_pos[env_id_single]
            joint_vel = rob.data.default_joint_vel[env_id_single]
            default_root_state = rob.data.default_root_state[env_id_single].clone()

            start_x = env_origins[env_idx, 0] + grid_x
            start_y = env_origins[env_idx, 1] + grid_y
            start_z = assigned_heights[j].item()

            default_root_state[:, 0] = start_x
            default_root_state[:, 1] = start_y
            default_root_state[:, 2] = start_z

            rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
            rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
            rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)

            goal_distance = torch.zeros(1, device=env.device).uniform_(
                2.0,
                env.cfg.curriculum.stage7_goal_distance,
            ).item()

            goal_angle = torch.zeros(1, device=env.device).uniform_(
                0.0,
                2.0 * torch.pi,
            ).item()

            goal_offset_x = goal_distance * torch.cos(torch.tensor(goal_angle, device=env.device))
            goal_offset_y = goal_distance * torch.sin(torch.tensor(goal_angle, device=env.device))

            goal_x = start_x + goal_offset_x
            goal_y = start_y + goal_offset_y

            goal_z_noise = torch.zeros(1, device=env.device).uniform_(-0.3, 0.3).item()
            goal_z = torch.clamp(
                torch.tensor(start_z + goal_z_noise, device=env.device),
                min=min_height,
                max=max_height,
            ).item()

            env._desired_pos_w[env_id_single, j, 0] = goal_x
            env._desired_pos_w[env_id_single, j, 1] = goal_y
            env._desired_pos_w[env_id_single, j, 2] = goal_z


def set_swarm_gravity_positions(env, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
    """Scatter-spawn (same pattern as set_stage2_positions/set_singlegoal_positions), then
    write ONE shared random target per env into every drone's _desired_pos_w -- no
    Hungarian assignment, no fixed formation shape. Agents self-organize around the
    target via the stage-8 reward's attraction + inter-agent repulsion terms.
    """
    num_reset_envs = len(env_ids)

    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(env.num_drones, dtype=torch.float32))))
    spacing = torch.zeros(1, device=env.device).uniform_(
        env.cfg.curriculum.spawn_grid_spacing_range[0],
        env.cfg.curriculum.spawn_grid_spacing_range[1],
    )

    min_height = env.cfg.curriculum.goal_height_range[0]
    max_height = env.cfg.curriculum.goal_height_range[1]
    spawn_lo, spawn_hi = env.cfg.curriculum.stage8_spawn_height_range
    base_height = torch.zeros(1, device=env.device).uniform_(spawn_lo, spawn_hi).item()

    for env_idx in range(num_reset_envs):
        env_id_single = env_ids[env_idx].unsqueeze(0)
        perm = torch.randperm(env.num_drones, device=env.device)

        heights = torch.arange(env.num_drones, device=env.device, dtype=torch.float32)
        heights = base_height + heights * env.cfg.curriculum.stage2_zdist_xy_plane
        heights = torch.clamp(heights, min=min_height, max=max_height)
        height_perm = torch.randperm(env.num_drones, device=env.device)
        assigned_heights = heights[height_perm]

        spawn_xy = torch.zeros(env.num_drones, 2, device=env.device)
        for j, rob in enumerate(env._robots):
            grid_idx = perm[j].item()
            grid_x = (grid_idx % grid_size) * spacing - (grid_size * spacing / 2.0)
            grid_y = (grid_idx // grid_size) * spacing - (grid_size * spacing / 2.0)

            joint_pos = rob.data.default_joint_pos[env_id_single]
            joint_vel = rob.data.default_joint_vel[env_id_single]
            default_root_state = rob.data.default_root_state[env_id_single].clone()

            start_x = env_origins[env_idx, 0] + grid_x
            start_y = env_origins[env_idx, 1] + grid_y
            start_z = assigned_heights[j].item()
            spawn_xy[j, 0] = start_x
            spawn_xy[j, 1] = start_y

            default_root_state[:, 0] = start_x
            default_root_state[:, 1] = start_y
            default_root_state[:, 2] = start_z

            rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
            rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
            rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)

        # One shared target for the whole swarm in this env -- offset from the swarm's
        # own spawn centroid so the task always requires real travel.
        centroid_xy = spawn_xy.mean(dim=0)
        target_distance = torch.zeros(1, device=env.device).uniform_(
            2.0, env.cfg.curriculum.stage8_target_distance,
        ).item()
        target_angle = torch.zeros(1, device=env.device).uniform_(0.0, 2.0 * torch.pi).item()
        target_x = centroid_xy[0] + target_distance * torch.cos(torch.tensor(target_angle, device=env.device))
        target_y = centroid_xy[1] + target_distance * torch.sin(torch.tensor(target_angle, device=env.device))
        target_z = torch.zeros(1, device=env.device).uniform_(
            env.cfg.curriculum.stage8_target_height_range[0],
            env.cfg.curriculum.stage8_target_height_range[1],
        ).item()

        env._desired_pos_w[env_id_single, :, 0] = target_x
        env._desired_pos_w[env_id_single, :, 1] = target_y
        env._desired_pos_w[env_id_single, :, 2] = target_z


def set_swarm_gravity_rm_positions(env, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
    """Stage 10 (SwarmGravityRM): identical spawn/target setup to set_swarm_gravity_positions
    (reused verbatim), then resets the per-agent RM-phase/slot-assignment tensors and
    samples a fresh random 3D rotation for this episode's packing-slot arrangement --
    see torchrl_swarm_env.py's stage-10 tensor allocations and termination.py's stage-10
    transition block for how these are consumed.
    """
    set_swarm_gravity_positions(env, env_ids, env_origins)

    env._swarm_rm_phase[env_ids, :] = 0
    env._entry_order[env_ids] = 0
    env._assigned_slot[env_ids, :] = -1

    num_reset_envs = len(env_ids)
    # Random 3D rotation per env: QR-decompose a random Gaussian matrix to get a uniform
    # orthonormal basis, then fix determinant to +1 (rotation, not reflection) -- same
    # "prevent overfitting to one fixed arrangement" reasoning as
    # get_inverted_v_formation's randomize_heading, extended from yaw-only to full 3D
    # since packing slots aren't planar.
    random_mat = torch.randn(num_reset_envs, 3, 3, device=env.device)
    q, r = torch.linalg.qr(random_mat)
    diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q = q * diag_sign.unsqueeze(-2)
    det = torch.linalg.det(q)
    q[:, :, -1] *= torch.sign(det).unsqueeze(-1)

    env._episode_slot_rotation[env_ids] = q


def set_stage3_positions(env, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
    """Set individual obstacle course navigation with waypoint-based goals.

    Each agent gets its own lane with 3 waypoints in a zig-zag pattern
    through obstacles.
    """
    num_reset_envs = len(env_ids)
    offset_x, offset_y = (0, 0)

    params = env.cfg.curriculum.get_stage3_params()

    min_height = env.cfg.curriculum.goal_height_range[0]
    max_height = env.cfg.curriculum.goal_height_range[1]

    spawn_x = -torch.zeros(num_reset_envs, device=env.device).uniform_(0.5, 1.0)

    for env_idx in range(num_reset_envs):
        env_id_int = env_ids[env_idx].item()
        env_id_single = env_ids[env_idx].unsqueeze(0)

        perm = torch.randperm(env.num_drones, device=env.device)
        spawn_lo, spawn_hi = env.cfg.curriculum.stage3_spawn_height_range
        base_height = torch.zeros(1, device=env.device).uniform_(spawn_lo, spawn_hi).item()

        for j, rob in enumerate(env._robots):
            agent_lane = perm[j].item()
            lane_center_y = offset_y + (agent_lane - (env.num_drones - 1) / 2.0) * params["lane_width"]

            start_x = env_origins[env_idx, 0] + spawn_x[env_idx]
            start_y = env_origins[env_idx, 1] + lane_center_y
            start_z = base_height

            joint_pos = rob.data.default_joint_pos[env_id_single]
            joint_vel = rob.data.default_joint_vel[env_id_single]
            default_root_state = rob.data.default_root_state[env_id_single].clone()

            default_root_state[:, 0] = start_x
            default_root_state[:, 1] = start_y
            default_root_state[:, 2] = start_z

            rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
            rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
            rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)

            for wp_idx in range(3):
                obs_x = env_origins[env_idx, 0] + offset_x + params["course_start_x"] + wp_idx * params["obstacle_spacing_x"]

                if wp_idx == 0:
                    obs_y = env_origins[env_idx, 1] + lane_center_y
                elif wp_idx == 1:
                    obs_y = env_origins[env_idx, 1] + lane_center_y - params["lateral_offset"]
                else:
                    obs_y = env_origins[env_idx, 1] + lane_center_y + params["lateral_offset"]

                waypoint_x = obs_x + params["waypoint_distance_behind"]
                waypoint_y = obs_y

                z_variation = torch.zeros(1, device=env.device).uniform_(-0.2, 0.2).item()
                waypoint_z = torch.clamp(
                    torch.tensor(base_height + z_variation, device=env.device),
                    min=min_height,
                    max=max_height,
                ).item()

                env._waypoint_paths[env_id_int, j, wp_idx, 0] = waypoint_x
                env._waypoint_paths[env_id_int, j, wp_idx, 1] = waypoint_y
                env._waypoint_paths[env_id_int, j, wp_idx, 2] = waypoint_z

            env._current_waypoint_idx[env_id_int, j] = 0
            env._desired_pos_w[env_id_single, j, :] = env._waypoint_paths[env_id_int, j, 0, :]


def set_stage4_positions(env, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
    """Set swarm navigation goals with formation aligned to travel direction.

    Agents start in an inverted-V formation and must navigate as a group
    to a translated goal while maintaining formation.
    """
    from .formation import get_inverted_v_formation

    num_reset_envs = len(env_ids)

    spawn_heights = torch.zeros(num_reset_envs, device=env.device).uniform_(
        env.cfg.curriculum.stage4_spawn_height_range[0],
        env.cfg.curriculum.stage4_spawn_height_range[1],
    )

    formation_positions = get_inverted_v_formation(env, env_ids, env_origins, spawn_heights)

    for j, rob in enumerate(env._robots):
        joint_pos = rob.data.default_joint_pos[env_ids]
        joint_vel = rob.data.default_joint_vel[env_ids]
        default_root_state = rob.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = formation_positions[:, j, :]

        rob.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    for env_idx in range(num_reset_envs):
        env_id_single = env_ids[env_idx]

        translation_distance = torch.zeros(1, device=env.device).uniform_(
            env.cfg.curriculum.swarm_translation_distance_range[0],
            env.cfg.curriculum.swarm_translation_distance_range[1],
        ).item()

        translation_angle = torch.zeros(1, device=env.device).uniform_(0.0, 2.0 * torch.pi).item()

        translation_x = translation_distance * torch.cos(torch.tensor(translation_angle, device=env.device))
        translation_y = translation_distance * torch.sin(torch.tensor(translation_angle, device=env.device))

        rotation_angle = translation_angle

        cos_theta = torch.cos(torch.tensor(rotation_angle, device=env.device))
        sin_theta = torch.sin(torch.tensor(rotation_angle, device=env.device))

        formation_center = formation_positions[env_idx].mean(dim=0)
        # Small random altitude delta so all drones in the formation share one goal z.
        # Previously this was "spawn_h + noise" which double-counted spawn height
        # (start_pos[2] already equals spawn_h), pushing goal_z to 2×spawn_h.
        goal_z_delta = torch.zeros(1, device=env.device).uniform_(-0.5, 0.5).item()

        for j in range(env.num_drones):
            start_pos = formation_positions[env_idx, j]
            relative_pos = start_pos[:2] - formation_center[:2]

            rotated_x = cos_theta * relative_pos[0] - sin_theta * relative_pos[1]
            rotated_y = sin_theta * relative_pos[0] + cos_theta * relative_pos[1]

            goal_x = formation_center[0] + translation_x + rotated_x
            goal_y = formation_center[1] + translation_y + rotated_y
            goal_z = start_pos[2] + goal_z_delta

            env._desired_pos_w[env_id_single, j, 0] = goal_x
            env._desired_pos_w[env_id_single, j, 1] = goal_y
            env._desired_pos_w[env_id_single, j, 2] = goal_z


def set_stage5_positions(env, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
    """Set swarm waypoint navigation through stacked X obstacle pattern.

    Swarm navigates through 3 waypoints placed at the centers of X pattern gaps.
    Formation is rotated 90 degrees to face the +Y direction (toward obstacles).
    """
    from .formation import get_inverted_v_formation

    num_reset_envs = len(env_ids)

    y_offset = env.cfg.curriculum.stage5_obsy_offset

    spawn_heights = torch.zeros(num_reset_envs, device=env.device).uniform_(
        env.cfg.curriculum.stage5_spawn_height_range[0],
        env.cfg.curriculum.stage5_spawn_height_range[1],
    )

    dist_y_from_spawn_swarm = torch.zeros(num_reset_envs, device=env.device).uniform_(0.8, 1.5)

    # Start positions: inverted V formation rotated to face +Y
    offset_origins = env_origins.clone()

    formation_positions = get_inverted_v_formation(env, env_ids, offset_origins, spawn_heights)

    # Rotate formation 90 degrees counterclockwise to face +Y direction
    for env_idx in range(num_reset_envs):
        formation_center = formation_positions[env_idx].mean(dim=0)

        for j in range(env.num_drones):
            relative_pos = formation_positions[env_idx, j, :2] - formation_center[:2]
            rotated_x = -relative_pos[1]
            rotated_y = -relative_pos[0]
            formation_positions[env_idx, j, 0] = formation_center[0] + rotated_x
            formation_positions[env_idx, j, 1] = formation_center[1] + rotated_y

    for j, rob in enumerate(env._robots):
        joint_pos = rob.data.default_joint_pos[env_ids]
        joint_vel = rob.data.default_joint_vel[env_ids]
        default_root_state = rob.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = formation_positions[:, j, :]

        rob.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    for env_idx in range(num_reset_envs):
        env_id_int = env_ids[env_idx].item()

        env_dist_y = dist_y_from_spawn_swarm[env_idx].item()

        base_height = spawn_heights[env_idx].item()
        swarm_goal_height = torch.zeros(1, device=env.device).uniform_(-0.5, 0.5).item()

        # Waypoint 1: Gap in bottom X
        wp1_x = env_origins[env_idx, 0]
        wp1_y = env_origins[env_idx, 1] + env_dist_y + 0.75 * y_offset
        wp1_z = base_height + swarm_goal_height

        env._swarm_waypoint_paths[env_id_int, 0, 0] = wp1_x
        env._swarm_waypoint_paths[env_id_int, 0, 1] = wp1_y
        env._swarm_waypoint_paths[env_id_int, 0, 2] = wp1_z

        # Waypoint 2: Gap in middle
        wp2_x = env_origins[env_idx, 0]
        wp2_y = env_origins[env_idx, 1] + env_dist_y + 1.25 * y_offset
        wp2_z = base_height + swarm_goal_height

        env._swarm_waypoint_paths[env_id_int, 1, 0] = wp2_x
        env._swarm_waypoint_paths[env_id_int, 1, 1] = wp2_y
        env._swarm_waypoint_paths[env_id_int, 1, 2] = wp2_z

        # Waypoint 3: Final position beyond top X
        wp3_x = env_origins[env_idx, 0]
        wp3_y = env_origins[env_idx, 1] + env_dist_y + 2.5 * y_offset
        wp3_z = base_height + swarm_goal_height

        env._swarm_waypoint_paths[env_id_int, 2, 0] = wp3_x
        env._swarm_waypoint_paths[env_id_int, 2, 1] = wp3_y
        env._swarm_waypoint_paths[env_id_int, 2, 2] = wp3_z

        env._current_swarm_waypoint_idx[env_id_int] = 0

    # Set initial goals (formation around first waypoint)
    for env_idx in range(num_reset_envs):
        env_id_single = env_ids[env_idx]
        env_id_int = env_id_single.item()

        swarm_target = env._swarm_waypoint_paths[env_id_int, 0, :]
        formation_center = formation_positions[env_idx].mean(dim=0)

        for j in range(env.num_drones):
            start_pos = formation_positions[env_idx, j]
            relative_pos = start_pos - formation_center
            goal_pos = swarm_target + relative_pos

            env._desired_pos_w[env_id_single, j, 0] = goal_pos[0]
            env._desired_pos_w[env_id_single, j, 1] = goal_pos[1]
            env._desired_pos_w[env_id_single, j, 2] = goal_pos[2]


def set_formation_positions(env, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
    """Scatter-spawn drones randomly, then assign each to the V-formation slot that
    minimizes total swarm travel distance (Hungarian / linear-sum assignment).

    Decentralized execution is preserved: the assignment is solved centrally here at
    reset time (same category as any other env-side goal generation), but each agent
    only ever observes its own assigned slot through the existing desired_pos_b field
    in _get_observations() -- never the assignment matrix or other agents' targets.
    """
    from scipy.optimize import linear_sum_assignment

    from .formation import get_inverted_v_formation

    num_reset_envs = len(env_ids)
    cfg_c = env.cfg.curriculum

    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(env.num_drones, dtype=torch.float32))))
    spacing = torch.zeros(1, device=env.device).uniform_(
        cfg_c.stage6_scatter_spacing_range[0],
        cfg_c.stage6_scatter_spacing_range[1],
    )

    spawn_lo, spawn_hi = cfg_c.stage6_spawn_height_range
    target_lo, target_hi = cfg_c.stage6_target_height_range

    # Target V-formation slots: one shared target altitude per env, independent of the
    # scattered spawn heights below, so the assignment problem is non-trivial in 3D.
    target_heights = torch.zeros(num_reset_envs, device=env.device).uniform_(target_lo, target_hi)
    formation_slots = get_inverted_v_formation(
        env, env_ids, env_origins, target_heights,
        randomize_heading=cfg_c.stage6_randomize_heading,
    )  # (num_reset_envs, num_drones, 3)

    for env_idx in range(num_reset_envs):
        env_id_single = env_ids[env_idx].unsqueeze(0)

        # Randomized grid scatter: guarantees minimum separation without rejection sampling.
        perm = torch.randperm(env.num_drones, device=env.device)
        heights = torch.zeros(env.num_drones, device=env.device).uniform_(spawn_lo, spawn_hi)
        spawn_positions = torch.zeros(env.num_drones, 3, device=env.device)

        for j in range(env.num_drones):
            grid_idx = perm[j].item()
            grid_x = (grid_idx % grid_size) * spacing - (grid_size * spacing / 2.0)
            grid_y = (grid_idx // grid_size) * spacing - (grid_size * spacing / 2.0)
            spawn_positions[j, 0] = env_origins[env_idx, 0] + grid_x
            spawn_positions[j, 1] = env_origins[env_idx, 1] + grid_y
            spawn_positions[j, 2] = heights[j]

        # Hungarian assignment: minimize total distance from scattered spawn to V-slots.
        cost_matrix = torch.cdist(
            spawn_positions.unsqueeze(0), formation_slots[env_idx].unsqueeze(0)
        ).squeeze(0).cpu().numpy()
        _, col_ind = linear_sum_assignment(cost_matrix)

        for j, rob in enumerate(env._robots):
            joint_pos = rob.data.default_joint_pos[env_id_single]
            joint_vel = rob.data.default_joint_vel[env_id_single]
            default_root_state = rob.data.default_root_state[env_id_single].clone()
            default_root_state[:, :3] = spawn_positions[j]

            rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
            rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
            rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)

            assigned_slot = formation_slots[env_idx, col_ind[j]]
            env._desired_pos_w[env_id_single, j, :] = assigned_slot
            env._assigned_slot_idx[env_id_single, j] = int(col_ind[j])
