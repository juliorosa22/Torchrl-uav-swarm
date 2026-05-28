"""Obstacle building and position collection for UAV swarm curriculum stages.

Handles construction of obstacle courses for Stage 3 (per-agent zig-zag lanes)
and Stage 5 (stacked-X pattern for swarm navigation).
"""

import torch
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg


def build_stage3_obstacles_at_origin(env) -> None:
    """Build stage 3 obstacles at environment origin (0, 0).

    Creates a zig-zag obstacle course with 3 obstacles per agent lane.
    """
    params = env.cfg.curriculum.get_stage3_params()
    obstacle_size = env.cfg.curriculum.obstacles_size
    base_height = obstacle_size[2] / 2.0

    global_wall_idx = 0

    for agent_idx in range(env.num_drones):
        lane_center_y = (agent_idx - (env.num_drones - 1) / 2.0) * params["lane_width"]

        for obs_idx in range(3):
            obs_x = params["course_start_x"] + obs_idx * params["obstacle_spacing_x"]

            # Zig-zag pattern
            if obs_idx == 0:
                obs_y = lane_center_y
            elif obs_idx == 1:
                obs_y = lane_center_y - params["lateral_offset"]
            else:
                obs_y = lane_center_y + params["lateral_offset"]

            obs_z = base_height

            wall_path = f"/World/envs/env_0/obstacles/wall_{global_wall_idx}"
            wall_cfg = CuboidCfg(
                size=obstacle_size,
                rigid_props=RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.1, 0.1),
                    roughness=0.4,
                    metallic=0.0,
                ),
            )
            wall_cfg.func(wall_path, wall_cfg, translation=(obs_x, obs_y, obs_z))
            global_wall_idx += 1

    _collect_obstacle_positions_stage3(env)


def build_stage5_obstacles_at_origin(env) -> None:
    """Build stage 5 obstacles at environment origin (0, 0).

    Creates a stacked-X pattern of 8 vertical walls for swarm navigation.
    """
    cfg = env.cfg.curriculum

    x_offset = cfg.stage5_obsx_offset
    y_offset = cfg.stage5_obsy_offset
    obstacle_size = cfg.obstacles_size
    # Inverted dimensions for vertical walls relative to Y-axis travel
    obstacle_size = (obstacle_size[1], obstacle_size[0], obstacle_size[2])
    base_height = obstacle_size[2] / 2.0

    dist_from_spawn_swarm = cfg.dist_from_spawn_swarm
    base_y = dist_from_spawn_swarm
    base_x = 0.0

    wall_positions = [
        (base_x - x_offset, base_y, base_height),
        (base_x + x_offset, base_y, base_height),
        (base_x, base_y + 0.5 * y_offset, base_height),
        (base_x - x_offset, base_y + y_offset, base_height),
        (base_x + x_offset, base_y + y_offset, base_height),
        (base_x, base_y + 1.5 * y_offset, base_height),
        (base_x - x_offset, base_y + 2 * y_offset, base_height),
        (base_x + x_offset, base_y + 2 * y_offset, base_height),
    ]

    stage5_start_idx = env.num_drones * 3

    for local_idx, (wall_x, wall_y, wall_z) in enumerate(wall_positions):
        global_wall_idx = stage5_start_idx + local_idx

        wall_path = f"/World/envs/env_0/obstacles/wall_{global_wall_idx}"
        wall_cfg = CuboidCfg(
            size=obstacle_size,
            rigid_props=RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.1, 0.9),
                roughness=0.4,
                metallic=0.0,
            ),
        )
        wall_cfg.func(wall_path, wall_cfg, translation=(wall_x, wall_y, wall_z))

    _collect_obstacle_positions_stage5(env)


def _collect_obstacle_positions_stage3(env) -> None:
    """Collect Stage 3 obstacle positions using shared parameters."""
    obstacle_positions = []

    params = env.cfg.curriculum.get_stage3_params()
    obstacle_size = env.cfg.curriculum.obstacles_size
    base_height = obstacle_size[2] / 2.0

    for agent_idx in range(env.num_drones):
        lane_center_y = (agent_idx - (env.num_drones - 1) / 2.0) * params["lane_width"]

        for obs_idx in range(3):
            obs_x = params["course_start_x"] + obs_idx * params["obstacle_spacing_x"]

            if obs_idx == 0:
                obs_y = lane_center_y
            elif obs_idx == 1:
                obs_y = lane_center_y - params["lateral_offset"]
            else:
                obs_y = lane_center_y + params["lateral_offset"]

            obs_z = base_height
            obstacle_positions.append([obs_x, obs_y, obs_z])

    env._obstacle_positions = torch.tensor(
        obstacle_positions,
        dtype=torch.float32,
        device=env.device,
    )

    print(f"[INFO] Collected {len(obstacle_positions)} Stage 3 obstacle positions")


def _collect_obstacle_positions_stage5(env) -> None:
    """Collect Stage 5 obstacle positions matching BUILD parameters."""
    obstacle_positions = []

    cfg = env.cfg.curriculum
    x_offset = cfg.stage5_obsx_offset
    y_offset = cfg.stage5_obsy_offset
    base_height = 4.0
    dist_from_spawn_swarm = cfg.dist_from_spawn_swarm
    base_y = dist_from_spawn_swarm
    base_x = 0.0

    wall_positions = [
        (base_x - x_offset, base_y, base_height),
        (base_x + x_offset, base_y, base_height),
        (base_x, base_y + 0.5 * y_offset, base_height),
        (base_x - x_offset, base_y + y_offset, base_height),
        (base_x + x_offset, base_y + y_offset, base_height),
        (base_x, base_y + 1.5 * y_offset, base_height),
        (base_x - x_offset, base_y + 2 * y_offset, base_height),
        (base_x + x_offset, base_y + 2 * y_offset, base_height),
    ]

    obstacle_positions.extend(wall_positions)

    env._obstacle_positions = torch.tensor(
        obstacle_positions,
        dtype=torch.float32,
        device=env.device,
    )
