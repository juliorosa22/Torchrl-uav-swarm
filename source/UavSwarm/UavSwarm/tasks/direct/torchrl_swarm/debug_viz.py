"""Debug visualization markers for UAV swarm environments.

Manages goal position markers, swarm centroid marker, and stage-specific
waypoint visualizations (stages 3 and 5).
"""

from isaaclab.markers import VisualizationMarkers, SPHERE_MARKER_CFG
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG


def set_debug_vis_impl(env, debug_vis: bool) -> None:
    """Initialize or hide all debug visualization markers.

    Creates yellow sphere markers for goals, blue arrow for swarm centroid,
    and cuboid markers for stage 3/5 waypoints.
    """
    if debug_vis:
        # Goal position markers (yellow spheres)
        if not hasattr(env, "goal_pos_visualizers"):
            env.goal_pos_visualizers = []
            for i in range(env.num_drones):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = 0.05
                marker_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                marker_cfg.prim_path = f"/Visuals/Command/goal_position_{i}"
                env.goal_pos_visualizers.append(VisualizationMarkers(marker_cfg))

        for viz in env.goal_pos_visualizers:
            viz.set_visibility(True)

        # Swarm centroid marker (blue arrow)
        if not hasattr(env, "centroid_visualizer"):
            centroid_marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
            centroid_marker_cfg.prim_path = "/Visuals/Command/swarm_centroid"
            env.centroid_visualizer = VisualizationMarkers(centroid_marker_cfg)

        if hasattr(env, "centroid_visualizer"):
            env.centroid_visualizer.set_visibility(True)

        # Stage 3: Individual waypoint markers (green cuboids)
        if not hasattr(env, "stage3_waypoint_visualizers"):
            env.stage3_waypoint_visualizers = []
            for i in range(env.num_drones):
                for wp_idx in range(env.num_waypoints_per_agent):
                    marker_cfg = CUBOID_MARKER_CFG.copy()
                    marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                    marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                    marker_cfg.prim_path = f"/Visuals/Command/stage3_waypoint_agent{i}_wp{wp_idx}"
                    env.stage3_waypoint_visualizers.append(VisualizationMarkers(marker_cfg))

        # Stage 5: Swarm waypoint markers (green cuboids)
        if not hasattr(env, "stage5_waypoint_visualizers"):
            env.stage5_waypoint_visualizers = []
            for wp_idx in range(env.num_swarm_waypoints):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.3, 0.3, 0.3)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                marker_cfg.prim_path = f"/Visuals/Command/stage5_swarm_waypoint_{wp_idx}"
                env.stage5_waypoint_visualizers.append(VisualizationMarkers(marker_cfg))

    else:
        if hasattr(env, "goal_pos_visualizers"):
            for viz in env.goal_pos_visualizers:
                viz.set_visibility(False)

        if hasattr(env, "centroid_visualizer"):
            env.centroid_visualizer.set_visibility(False)

        if hasattr(env, "stage3_waypoint_visualizers"):
            for viz in env.stage3_waypoint_visualizers:
                viz.set_visibility(False)

        if hasattr(env, "stage5_waypoint_visualizers"):
            for viz in env.stage5_waypoint_visualizers:
                viz.set_visibility(False)


def debug_vis_callback(env, _event) -> None:
    """Update debug visualization markers each physics step.

    Shows/hides markers based on current curriculum stage:
    - Goal markers: always visible
    - Centroid marker: stages 4-5 only
    - Stage 3 waypoints: stage 3 only
    - Stage 5 waypoints: stage 5 only
    """
    # Update goal position markers (always visible)
    if hasattr(env, "goal_pos_visualizers"):
        for i, viz in enumerate(env.goal_pos_visualizers):
            viz.visualize(env._desired_pos_w[:, i, :])

    # Update swarm centroid marker (stages 4 and 5 only)
    if hasattr(env, "centroid_visualizer"):
        if env.curriculum_stage in [4, 5]:
            env.centroid_visualizer.set_visibility(True)
            env.centroid_visualizer.visualize(env._swarm_centroid)
        else:
            env.centroid_visualizer.set_visibility(False)

    # Update stage 3 waypoint markers
    if hasattr(env, "stage3_waypoint_visualizers"):
        if env.curriculum_stage == 3:
            viz_idx = 0
            for agent_idx in range(env.num_drones):
                for wp_idx in range(env.num_waypoints_per_agent):
                    waypoint_positions = env._waypoint_paths[:, agent_idx, wp_idx, :]
                    env.stage3_waypoint_visualizers[viz_idx].set_visibility(True)
                    env.stage3_waypoint_visualizers[viz_idx].visualize(waypoint_positions)
                    viz_idx += 1
        else:
            for viz in env.stage3_waypoint_visualizers:
                viz.set_visibility(False)

    # Update stage 5 swarm waypoint markers
    if hasattr(env, "stage5_waypoint_visualizers"):
        if env.curriculum_stage == 5:
            for wp_idx in range(env.num_swarm_waypoints):
                waypoint_positions = env._swarm_waypoint_paths[:, wp_idx, :]
                env.stage5_waypoint_visualizers[wp_idx].set_visibility(True)
                env.stage5_waypoint_visualizers[wp_idx].visualize(waypoint_positions)
        else:
            for viz in env.stage5_waypoint_visualizers:
                viz.set_visibility(False)
