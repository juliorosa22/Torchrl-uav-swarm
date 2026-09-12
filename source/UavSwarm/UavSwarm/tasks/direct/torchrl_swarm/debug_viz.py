"""Debug visualization markers for UAV swarm environments.

Manages goal position markers, swarm centroid marker, and stage-specific
waypoint visualizations (stages 3 and 5).
"""

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, SPHERE_MARKER_CFG
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG
from isaaclab.utils.math import quat_from_angle_axis


def _build_containment_wireframe(radius: float, device: str, num_segments: int = 28) -> tuple[torch.Tensor, torch.Tensor]:
    """Local (center-relative) positions/orientations for a wireframe sphere: 3 mutually
    perpendicular great-circle rings, each built from short cylinder segments tangent to
    the circle -- solid geometry (unlike a transparent sphere, this actually renders
    correctly in an offscreen RTX capture -- see set_debug_vis_impl's note) with visible
    gaps between segments so agents crossing into the volume stay visible.

    Returns (positions (3*num_segments, 3), quaternions (3*num_segments, 4)), to be
    translated by the current target position each step (see debug_vis_callback) -- the
    pattern's shape never changes, only where it's centered.
    """
    theta = torch.linspace(0, 2 * math.pi, num_segments + 1, device=device)[:-1]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(num_segments, 3)

    all_pos, all_quat = [], []
    # (u, v) basis per plane: XY, XZ, YZ -- point(t)=R*(cos*u+sin*v), tangent(t) ~ -sin*u+cos*v
    for u, v in [
        (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])),
        (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),
        (torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),
    ]:
        u, v = u.to(device), v.to(device)
        pos = radius * (cos_t.unsqueeze(-1) * u + sin_t.unsqueeze(-1) * v)
        tangent = -sin_t.unsqueeze(-1) * u + cos_t.unsqueeze(-1) * v
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        axis = torch.cross(z_axis, tangent, dim=-1)
        axis_norm = axis.norm(dim=-1, keepdim=True)
        angle = torch.acos((z_axis * tangent).sum(-1).clamp(-1.0, 1.0))
        safe_axis = torch.where(
            axis_norm > 1e-6, axis / axis_norm.clamp(min=1e-6),
            torch.tensor([1.0, 0.0, 0.0], device=device).expand(num_segments, 3),
        )
        all_pos.append(pos)
        all_quat.append(quat_from_angle_axis(angle, safe_axis))

    return torch.cat(all_pos, dim=0), torch.cat(all_quat, dim=0)


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

        # Stage 8/9: containment-boundary marker (wireframe sphere, cyan) -- radius
        # derived the same way as the success condition
        # (CurriculumCfg.get_containment_radius), so what's drawn is exactly the volume
        # _check_swarm_gravity_reached checks against, not a separately-tuned cosmetic
        # value. Built from solid cylinder segments (3 perpendicular rings), not a
        # transparent sphere -- neither PreviewSurfaceCfg.opacity (only affects
        # interactive rendering per its own docstring) nor GlassMdlCfg (renders flat
        # black/white here -- no skybox/environment texture in this scene for it to
        # refract) produced real translucency in an offscreen RTX capture; solid
        # wireframe geometry sidesteps the problem entirely and still lets agents be
        # seen crossing into the volume through the gaps between segments.
        if env.curriculum_stage in (8, 9) and not hasattr(env, "containment_sphere_visualizer"):
            radius = env.cfg.curriculum.get_containment_radius(
                env.num_drones, env.cfg.swarm_cfg.min_safe_distance
            )
            num_segments = 28
            arc_len = 2 * math.pi * radius / num_segments
            containment_marker_cfg = VisualizationMarkersCfg(
                markers={
                    "segment": sim_utils.CylinderCfg(
                        radius=0.03, height=arc_len * 0.85,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                    ),
                },
                prim_path="/Visuals/Command/containment_sphere",
            )
            env.containment_sphere_visualizer = VisualizationMarkers(containment_marker_cfg)
            env._containment_wireframe_local_pos, env._containment_wireframe_local_quat = (
                _build_containment_wireframe(radius, str(env.device), num_segments)
            )

        if hasattr(env, "containment_sphere_visualizer"):
            env.containment_sphere_visualizer.set_visibility(True)

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

        if hasattr(env, "containment_sphere_visualizer"):
            env.containment_sphere_visualizer.set_visibility(False)

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

    # Update containment-boundary wireframe (stages 8/9 only) -- every agent shares the
    # same target (_desired_pos_w is identical across agents for these stages), so
    # agent-0's slot is the shared target position. The wireframe pattern itself is
    # fixed (computed once in set_debug_vis_impl); only its center translates.
    if hasattr(env, "containment_sphere_visualizer"):
        if env.curriculum_stage in (8, 9):
            env.containment_sphere_visualizer.set_visibility(True)
            target = env._desired_pos_w[:, 0, :]  # (num_envs, 3)
            local_pos = env._containment_wireframe_local_pos  # (S, 3)
            local_quat = env._containment_wireframe_local_quat  # (S, 4)
            translations = (target.unsqueeze(1) + local_pos.unsqueeze(0)).reshape(-1, 3)
            quats = local_quat.unsqueeze(0).expand(target.shape[0], -1, -1).reshape(-1, 4)
            env.containment_sphere_visualizer.visualize(translations, quats)
        else:
            env.containment_sphere_visualizer.set_visibility(False)

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
