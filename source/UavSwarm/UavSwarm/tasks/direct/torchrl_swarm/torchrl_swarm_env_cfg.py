"""Configuration classes for UAV swarm RL environments.

Provides curriculum, reward machine, and swarm parameter configs,
plus a shared base environment config with FullTask/Baseline variants.
"""

from __future__ import annotations

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.utils import configclass

from .controller import ControllerCfg


class UavSwarmEnvWindow(BaseEnvWindow):
    """Window manager for the UAV Swarm environment."""

    def __init__(self, env, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", env)


@configclass
class CurriculumCfg:
    """Curriculum learning configuration for progressive skill acquisition."""

    active_stage: int = 5

    # Episode durations per stage
    stage1_episode_length_s: float = 60.0
    stage2_episode_length_s: float = 90.0
    stage3_episode_length_s: float = 240.0
    stage4_episode_length_s: float = 300.0
    stage5_episode_length_s: float = 300.0
    stage6_episode_length_s: float = 180.0
    stage7_episode_length_s: float = 90.0
    stage8_episode_length_s: float = 120.0

    # Stage 2 parameters
    stage2_goal_distance: float = 6.0
    stage2_zdist_xy_plane: float = 1.0

    # Stage 3 parameters
    stage3_params: dict = {
        "lane_width": 1.2,
        "obstacle_spacing_x": 2.5,
        "lateral_offset": 0.3,
        "course_start_x": 1.5,
        "waypoint_distance_behind": 1.0,
    }

    # Stage 4 parameters
    swarm_translation_distance_range: tuple = (2, 5)

    # Stage 5 parameters
    stage5_obsx_offset: float = 1.5
    stage5_obsy_offset: float = 2.0
    dist_from_spawn_swarm: float = 1.5

    # Common parameters
    spawn_height_range: tuple = (0.8, 1.2)
    spawn_grid_spacing_range: tuple = (0.5, 0.8)
    goal_height_range: tuple = (1.5, 6.0)
    max_obstacle_distance: float = 10.0
    obstacles_size: tuple = (0.15, 0.8, 8.0)

    # Stage 1 hover parameters
    # Spawn high enough that a random descending policy has several seconds of
    # margin before hitting min_flight_height (0.1m). Goal is offset above spawn
    # so difficulty is consistent regardless of spawn height.
    stage1_spawn_height_range: tuple = (1.5, 2.5)
    stage1_goal_height_delta_range: tuple = (0.5, 2.0)

    # Stage 2–5 spawn height ranges
    # All use (lo, hi) analogous to Stage 1: minimum raised above 1.5 m so
    # a random policy has ≥1 s of descent margin before min_flight_height=0.1 m.
    stage2_spawn_height_range: tuple = (2.0, 4.0)
    stage3_spawn_height_range: tuple = (2.0, 5.0)
    stage4_spawn_height_range: tuple = (2.0, 4.5)
    stage5_spawn_height_range: tuple = (2.0, 5.0)

    # Stage 6 parameters (formation-assignment scalability task)
    # Scatter spacing is deliberately wider than spawn_grid_spacing_range so drones start
    # meaningfully far from their eventual slot -- otherwise the assignment problem is
    # trivial and there's nothing for the Hungarian solve (or the policy) to do.
    stage6_scatter_spacing_range: tuple = (1.5, 3.0)
    stage6_spawn_height_range: tuple = (2.0, 4.5)
    # Target formation altitude, independent of scattered spawn heights.
    stage6_target_height_range: tuple = (1.5, 3.0)
    # Randomize the V-formation's world-frame heading each episode (see
    # formation.py::get_inverted_v_formation) instead of always pointing -X, so the policy
    # can't overfit to a single formation orientation.
    stage6_randomize_heading: bool = True
    # Diagnostic switch: when True, stage 6 uses get_formation_rewards_simple() instead of
    # get_formation_rewards() -- only the pose-distance term to the assigned slot, no
    # delta/alignment/safety/jerk terms. See rewards.py for why (training plateaued with
    # the full reward despite stable formation-holding).
    stage6_simple_reward: bool = False

    # Stage 7 parameters (single-goal navigation -- isolates the formation task's own
    # get_formation_rewards_simple/termination path from Hungarian/V-formation assignment,
    # to test whether the formation-assignment machinery itself matters).
    stage7_spawn_height_range: tuple = (2.0, 4.0)
    stage7_goal_distance: float = 6.0
    stage7_zdist_xy_plane: float = 1.0

    # Stage 8 parameters (swarm-gravity: one shared target, no fixed formation shape --
    # agents self-organize around it, attracted to the target and repelled from
    # neighbors closer than swarm_cfg.min_safe_distance = R_nh).
    stage8_spawn_height_range: tuple = (2.0, 4.0)
    stage8_target_distance: float = 6.0
    stage8_target_height_range: tuple = (1.5, 4.0)
    # R_gv: the closest agent must be within this distance of the target to count as
    # "arrived" -- with R_nh > 0 forcing separation, agents can't all sit on the target,
    # so this measures the swarm's leading edge, not the whole swarm.
    stage8_gravity_radius: float = 0.5
    # Packing-efficiency factor for the containment-sphere radius (see
    # CurriculumCfg.get_containment_radius): real swarms don't pack as tightly as
    # hexagonal close-packing (~0.74); 0.6 is a reasonable loose-swarm default.
    stage8_packing_density: float = 0.6

    def get_containment_radius(self, num_drones: int, min_safe_distance: float) -> float:
        """Sphere radius that must contain every agent for a stage-8 episode to count as
        successful (see docstring on stage8_packing_density). Derived from the
        equivalent-volume packing argument: N agents each need a non-overlapping ball of
        radius R_nh/2 (two such balls don't overlap iff centers are >= R_nh apart), so the
        containing sphere's volume must be >= N times that, inflated by 1/packing_density
        for imperfect packing:
            R_containment^3 = N * (R_nh/2)^3 / packing_density
        """
        r_nh_half = min_safe_distance / 2.0
        return r_nh_half * (num_drones / self.stage8_packing_density) ** (1.0 / 3.0)

    def get_episode_length(self) -> float:
        """Return episode length based on active stage."""
        stage_lengths = {
            1: self.stage1_episode_length_s,
            2: self.stage2_episode_length_s,
            3: self.stage3_episode_length_s,
            4: self.stage4_episode_length_s,
            5: self.stage5_episode_length_s,
            6: self.stage6_episode_length_s,
            7: self.stage7_episode_length_s,
            8: self.stage8_episode_length_s,
        }
        if self.active_stage not in stage_lengths:
            raise ValueError(f"Invalid active_stage: {self.active_stage}. Must be 1-8.")
        return stage_lengths[self.active_stage]

    def get_stage3_params(self) -> dict:
        return self.stage3_params.copy()


@configclass
class RewardMachineCfg:
    """Configuration for Reward Machine parameters."""

    # Switch thresholds and reward scales for each RM state
    exit_hover_altitude: float = 0.8
    enter_obstacle_avoidance_dist: float = 0.3
    enter_coop_moving_dist: float = 3

    # Reward Machine states parameters
    hover_min_altitude: float = 0.8
    close_obs_dist_thresh: float = 0.3
    num_rm_states: int = 4

    # Termination thresholds
    min_flight_height: float = 0.1
    max_flight_height: float = 15.0
    max_distance_from_origin: float = 15.0

    # Goal reaching thresholds (stage-dependent)
    hover_position_threshold: float = 0.15
    hover_velocity_threshold: float = 0.2
    goal_position_threshold: float = 0.15
    swarm_goal_threshold: float = 0.3

    # Base reward scales
    lin_vel_reward_scale: float = -0.01
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 5.0
    formation_penalty_scale: float = -0.3
    collision_penalty_scale: float = -5.0

    # RM state-aware bonus scales
    altitude_bonus_scale: float = 0.5
    obstacle_bonus_scale: float = 0.8
    neighbor_bonus_scale: float = 0.5
    state_progress_scale: float = 0.3

    # Safe distance thresholds
    safe_obstacle_distance: float = 0.4
    optimal_neighbor_distance: float = 1


@configclass
class SwarmParameterCfg:
    """Configuration for Swarm parameters."""

    # Swarm sensing parameters
    max_neighbor_distance: float = 10.0
    min_safe_distance: float = 1
    optimal_distance: float = 3.0
    max_formation_distance: float = 6.0

    # Inverted V formation parameters
    formation_base_separation = 0.8
    formation_v_angle_deg = 60.0
    formation_apex_offset = 0.0


@configclass
class BaseSwarmEnvCfg(DirectMARLEnvCfg):
    """Shared base configuration for UAV swarm environments.

    Subclasses set include_rm_in_obs to control whether the 4-dim RM
    state one-hot is included in per-agent observations.
    """

    include_rm_in_obs: bool = True

    # Episode / stepping
    episode_length_s = 30.0
    decimation = 2
    num_agents: int = 5
    max_num_agents: int = 20

    curriculum: CurriculumCfg = CurriculumCfg()
    reward_cfg: RewardMachineCfg = RewardMachineCfg()
    swarm_cfg: SwarmParameterCfg = SwarmParameterCfg()
    controller: ControllerCfg = ControllerCfg()

    # Observation / action / state dimensions
    # Base: 3+3+3+3+1+3+3+3+3+3 = 28, FullTask adds 4 RM one-hot = 32
    single_observation_space: int = 32
    single_action_space: int = 4
    state_space: int = 160  # num_agents * single_observation_space

    # Agent specs (computed at class-def time; configclass deep-copies per instance)
    possible_agents: list = [f"robot_{i}" for i in range(num_agents)]
    action_spaces: dict = {f"robot_{i}": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for i in range(num_agents)}
    observation_spaces: dict = {f"robot_{i}": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(32,)) for i in range(num_agents)}

    # Debug visualization
    debug_vis = True
    ui_window_class_type = UavSwarmEnvWindow

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground)

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=8.0, replicate_physics=True, clone_in_fabric=True
    )

    # Robot template
    robot_template: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).replace(
        spawn=CRAZYFLIE_CFG.spawn.replace(
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0),
                roughness=0.5,
                metallic=0.2,
            ),
            visual_material_path="/World/Looks/CrazyflieBlack",
        )
    )

    # Action -> force/torque conversion
    thrust_to_weight = 1.9
    moment_scale = 0.01


@configclass
class FullTaskUAVSwarmEnvCfg(BaseSwarmEnvCfg):
    """FullTask variant: includes RM state one-hot in observations (32-dim)."""

    include_rm_in_obs: bool = True
    single_observation_space: int = 32
    state_space: int = 160
    observation_spaces: dict = {f"robot_{i}": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(32,)) for i in range(5)}


@configclass
class BaselineUAVSwarmEnvCfg(BaseSwarmEnvCfg):
    """Baseline variant: no RM state in observations (28-dim)."""

    include_rm_in_obs: bool = False
    single_observation_space: int = 28
    state_space: int = 140
    observation_spaces: dict = {f"robot_{i}": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(28,)) for i in range(5)}


@configclass
class FormationUAVSwarmEnvCfg(BaseSwarmEnvCfg):
    """Formation-assignment scalability task: RM-free (baseline arm), single focused
    task -- scatter-spawn then converge to a Hungarian-assigned V-formation slot.

    Uses curriculum stage 6 purely as an internal dispatch value into the shared
    BaseSwarmEnv workflow; it is not part of the sequential 1-5 curriculum.
    """

    include_rm_in_obs: bool = False
    single_observation_space: int = 28
    state_space: int = 140
    observation_spaces: dict = {f"robot_{i}": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(28,)) for i in range(5)}

    curriculum: CurriculumCfg = CurriculumCfg(active_stage=6)


@configclass
class SingleGoalUAVSwarmEnvCfg(BaseSwarmEnvCfg):
    """Single-UAV point-to-target navigation: same 28-dim obs schema and
    get_formation_rewards_simple/_check_individual_goals_reached path as the Formation
    task, but goals are independently sampled per agent (no Hungarian/V-formation
    assignment). Isolates whether the formation-assignment machinery itself matters to
    the Formation task's convergence, vs. the shared obs/reward/controller/physics path.

    Uses curriculum stage 7 purely as an internal dispatch value, same pattern as
    Formation's stage 6. Defaults to num_agents=5 like every other cfg class here (baking
    num_agents=1 directly into the class body broke PhysX scene setup -- "Failed to get
    DOF positions from backend" -- for reasons not fully understood; the proven path is
    mappo_train.py's --num_agents CLI override, which patches an already-constructed
    5-agent cfg instance at runtime instead). Pass --num_agents 1 at launch for the actual
    single-UAV diagnostic.
    """

    include_rm_in_obs: bool = False
    single_observation_space: int = 28
    state_space: int = 140
    observation_spaces: dict = {f"robot_{i}": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(28,)) for i in range(5)}

    curriculum: CurriculumCfg = CurriculumCfg(active_stage=7)


@configclass
class SwarmGravityUAVSwarmEnvCfg(BaseSwarmEnvCfg):
    """Swarm-gravity task: one shared target position per env (not N per-agent slots) --
    agents are attracted to it and repelled from neighbors closer than
    swarm_cfg.min_safe_distance (R_nh), self-organizing into a cluster around the target
    with no fixed formation shape. Success requires every agent within a containment
    sphere (see CurriculumCfg.get_containment_radius) AND the closest agent within
    stage8_gravity_radius (R_gv) of the target.

    Uses curriculum stage 8 purely as an internal dispatch value, same pattern as
    Formation's stage 6 / SingleGoal's stage 7.
    """

    include_rm_in_obs: bool = False
    single_observation_space: int = 28
    state_space: int = 140
    observation_spaces: dict = {f"robot_{i}": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(28,)) for i in range(5)}

    curriculum: CurriculumCfg = CurriculumCfg(active_stage=8)
