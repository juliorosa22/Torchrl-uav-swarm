"""Shared base environment for UAV swarm curriculum training.

BaseSwarmEnv provides all common logic. FullTask and Baseline variants
differ only in their config class, which controls whether the 4-dim RM
state one-hot is included in per-agent observations.
"""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import subtract_frame_transforms

from .torchrl_swarm_env_cfg import (
    BaseSwarmEnvCfg,
    FullTaskUAVSwarmEnvCfg,
    BaselineUAVSwarmEnvCfg,
    FormationUAVSwarmEnvCfg,
    SingleGoalUAVSwarmEnvCfg,
    SwarmGravityUAVSwarmEnvCfg,
    SwarmGravityV2UAVSwarmEnvCfg,
)
from .controller import apply_controller
from .metrics import EpisodeMetrics
from .formation import compute_swarm_centroid, get_inverted_v_formation
from .sensing import ensure_cache_populated
from .rm_state import switch_rm_state
from .obstacles import build_stage3_obstacles_at_origin, build_stage5_obstacles_at_origin
from .curriculum import (
    set_stage1_positions,
    set_stage2_positions,
    set_stage3_positions,
    set_stage4_positions,
    set_stage5_positions,
    set_formation_positions,
    set_singlegoal_positions,
    set_swarm_gravity_positions,
)
from .termination import (
    get_dones,
    update_waypoint_goals,
    update_swarm_waypoint_goals,
)
from .rewards import get_rewards, get_formation_rewards, get_formation_rewards_simple, get_swarm_gravity_rewards
from .debug_viz import set_debug_vis_impl, debug_vis_callback


class BaseSwarmEnv(DirectMARLEnv):
    """Direct-style MARL environment with N Crazyflies per env.

    Actions: per-drone [vx_b, vy_b, vz_b, yaw_rate] in [-1,1] (geometric/pd_velocity)
             or [thrust, mx, my, mz] in [-1,1] (direct). Controlled by cfg.controller.type.
    Observations: 28-dim (Baseline) or 32-dim (FullTask, +4 RM one-hot)
    Rewards: energy-based with RM state-aware weighting + safety penalties
    """

    cfg: BaseSwarmEnvCfg

    def __init__(self, cfg: BaseSwarmEnvCfg, render_mode: str | None = None, **kwargs):
        self.num_drones = cfg.num_agents
        self.global_step = 0

        self._obstacles_built = False
        self.curriculum_stage = cfg.curriculum.active_stage

        cfg.episode_length_s = cfg.curriculum.get_episode_length()
        print(f"[INFO] Stage {cfg.curriculum.active_stage}: Episode length = {cfg.episode_length_s}s")

        self._robots = []
        self._body_ids = []
        self._obstacle_positions = None
        super().__init__(cfg, render_mode, **kwargs)

        # Post-init tensor allocations (device is now available)
        self._actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, self.num_drones, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, self.num_drones, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, self.num_drones, 3, device=self.device)
        # Stage 6 only: which V-formation slot (a stable geometric identity -- 0=apex,
        # increasing index = further out on a wing, see get_inverted_v_formation) each
        # drone was Hungarian-assigned to this episode. Drone index itself carries no
        # signal across episodes since the assignment is re-solved every reset; slot
        # index does, and is what lets _reset_idx's per-slot metrics distinguish "one
        # specific slot always lags" from "a random agent lags each episode".
        self._assigned_slot_idx = torch.zeros(self.num_envs, self.num_drones, dtype=torch.long, device=self.device)

        self._last_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_timed_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # RM state buffers: 0=H, 1=S, 2=C, 3=O
        self._rm_states = torch.zeros(self.num_envs, self.num_drones, dtype=torch.long, device=self.device)
        self._rm_state_names = ['H', 'S', 'C', 'O']

        # Cache buffers
        self._cached_obstacle_dists = torch.zeros(self.num_drones, self.num_envs, device=self.device)
        self._cached_obstacle_dir_b = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)
        self._cached_neighbor_rel_pos_b = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)
        self._cached_neighbor_rel_vel_b = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)
        self._cached_mean_neighbor_pos_b = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)
        self._cached_mean_neighbor_vel_b = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)
        self._prev_distances = torch.zeros(self.num_drones, self.num_envs, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)
        self._cache_valid = False

        self._metrics = EpisodeMetrics.create(self.num_envs, self.device)

        # Body indices and masses
        self._body_ids = [rob.find_bodies("body")[0] for rob in self._robots]
        masses = [rob.root_physx_view.get_masses()[0].sum() for rob in self._robots]
        self._masses = torch.tensor(masses, device=self.device).view(1, self.num_drones)
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weights = (self._masses * self._gravity_magnitude).squeeze(0)

        # Waypoint buffers
        self.num_waypoints_per_agent = 3
        self._waypoint_paths = torch.zeros(
            self.num_envs, self.num_drones, self.num_waypoints_per_agent, 3, device=self.device
        )
        self._current_waypoint_idx = torch.zeros(
            self.num_envs, self.num_drones, dtype=torch.long, device=self.device
        )

        self.num_swarm_waypoints = 3
        self._swarm_waypoint_paths = torch.zeros(
            self.num_envs, self.num_swarm_waypoints, 3, device=self.device
        )
        self._current_swarm_waypoint_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self._swarm_centroid = torch.zeros(self.num_envs, 3, device=self.device)
        self.waypoint_reach_threshold = 0.2
        self.swarm_waypoint_reach_threshold = 0.5

        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        """Setup the scene with terrain and N robots per environment."""
        for i in range(self.num_drones):
            robot_cfg: ArticulationCfg = self.cfg.robot_template.replace(
                prim_path=f"/World/envs/env_.*/Robot_{i}"
            ).replace(
                spawn=self.cfg.robot_template.spawn.replace(
                    visual_material_path=f"/World/Looks/Crazyflie_{i}"
                )
            )
            robot = Articulation(robot_cfg)
            self.scene.articulations[f"robot_{i}"] = robot
            self._robots.append(robot)

        if self.curriculum_stage == 3:
            print("[INFO] Building Stage 3 obstacles at origin...")
            build_stage3_obstacles_at_origin(self)
        elif self.curriculum_stage == 5:
            print("[INFO] Building Stage 5 obstacles at origin...")
            build_stage5_obstacles_at_origin(self)
        else:
            print(f"[INFO] Stage {self.curriculum_stage} has no obstacles")

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # RL workflow
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        """Convert actions to thrust and moments for each drone."""
        if self.curriculum_stage == 3:
            update_waypoint_goals(self)
        elif self.curriculum_stage == 5:
            update_swarm_waypoint_goals(self)

        if self.curriculum_stage in [4, 5, 6]:
            compute_swarm_centroid(self)

        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        switch_rm_state(self, all_positions)

        actions_list = [actions[f"robot_{i}"] for i in range(self.num_drones)]
        actions_tensor = torch.stack(actions_list, dim=1)
        self._actions = actions_tensor.clone().clamp(-1.0, 1.0)

        self._thrust, self._moment = apply_controller(self, self._actions)

    def _apply_action(self):
        """Apply forces and torques to each robot."""
        for j, rob in enumerate(self._robots):
            rob.set_external_force_and_torque(
                self._thrust[:, j, :, :],
                self._moment[:, j, :, :],
                body_ids=self._body_ids[j],
            )
        self._cache_valid = False

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments with curriculum-dependent goals and scene adjustments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

        # Log episodic metrics
        final_distances = []
        for j in range(self.num_drones):
            dist = torch.linalg.norm(
                self._desired_pos_w[env_ids, j, :] - self._robots[j].data.root_pos_w[env_ids],
                dim=1,
            )
            final_distances.append(dist)
        # (num_drones, num_reset_envs): per-agent distance to its own assigned slot.
        final_distances_stacked = torch.stack(final_distances)
        final_distance_to_goal = final_distances_stacked.mean()
        # Per-env closest/farthest agent, averaged over reset envs -- distinguishes a
        # uniform shortfall (min and max close together) from a split where some agents
        # converge and others don't (mean alone can't tell these apart).
        final_distance_to_goal_min = final_distances_stacked.min(dim=0)[0].mean()
        final_distance_to_goal_max = final_distances_stacked.max(dim=0)[0].mean()

        log_dict = self._metrics.to_log_dict(
            env_ids=env_ids,
            max_episode_length=self.max_episode_length_s,
            prefix="Episode_Reward",
        )

        # Rates (fraction of the envs resetting this call), not raw counts -- comparable
        # across calls regardless of how many envs happened to reset together, and directly
        # comparable to eval_formation_scalability.py's success_rate/collision_rate.
        n_reset = max(len(env_ids), 1)
        log_dict["Episode_Termination/died"] = torch.count_nonzero(self._last_terminated[env_ids]).item() / n_reset
        log_dict["Episode_Termination/time_out"] = torch.count_nonzero(self._last_timed_out[env_ids]).item() / n_reset

        if hasattr(self, '_termination_reasons'):
            log_dict["Episode_Termination/collision"] = torch.count_nonzero(
                self._termination_reasons['collision'][env_ids]
            ).item() / n_reset
            log_dict["Episode_Termination/out_of_bounds"] = torch.count_nonzero(
                self._termination_reasons['out_of_bounds'][env_ids]
            ).item() / n_reset
            log_dict["Episode_Termination/goal_reached"] = torch.count_nonzero(
                self._termination_reasons['goal_reached'][env_ids]
            ).item() / n_reset
            if 'inter_agent_collision' in self._termination_reasons:
                log_dict["Episode_Termination/inter_agent_collision"] = torch.count_nonzero(
                    self._termination_reasons['inter_agent_collision'][env_ids]
                ).item() / n_reset

        log_dict["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        log_dict["Metrics/final_distance_to_goal_min"] = final_distance_to_goal_min.item()
        log_dict["Metrics/final_distance_to_goal_max"] = final_distance_to_goal_max.item()
        log_dict["Metrics/curriculum_stage"] = self.curriculum_stage

        if self.curriculum_stage == 6:
            # Per-slot breakdown: distinguishes "one specific V-formation slot always
            # lags" from "a random agent lags each episode" -- slot index is a stable
            # geometric identity (0=apex, see get_inverted_v_formation) across episodes,
            # unlike drone index, which is reshuffled every reset by the Hungarian
            # assignment in set_formation_positions().
            slot_ids = self._assigned_slot_idx[env_ids].t()  # (num_drones, num_reset_envs)
            for slot in range(self.num_drones):
                slot_mask = slot_ids == slot
                if slot_mask.any():
                    log_dict[f"Metrics/final_distance_by_slot/slot_{slot}"] = (
                        final_distances_stacked[slot_mask].mean().item()
                    )
            # Per-drone (physical identity) breakdown, to separately rule out a
            # drone-specific issue (e.g. asymmetric mass/thrust) independent of slot.
            for j in range(self.num_drones):
                log_dict[f"Metrics/final_distance_by_drone/drone_{j}"] = (
                    final_distances_stacked[j].mean().item()
                )

        if self.curriculum_stage == 3:
            avg_waypoint_progress = self._current_waypoint_idx[env_ids].float().mean().item()
            log_dict["Metrics/avg_waypoint_progress"] = avg_waypoint_progress / self.num_waypoints_per_agent
        elif self.curriculum_stage == 5:
            avg_swarm_waypoint_progress = self._current_swarm_waypoint_idx[env_ids].float().mean().item()
            log_dict["Metrics/avg_swarm_waypoint_progress"] = avg_swarm_waypoint_progress / self.num_swarm_waypoints

        self.extras["log"] = log_dict

        self._metrics.reset(env_ids)
        self._cache_valid = False
        # NOTE: do NOT clear _last_terminated/_last_timed_out here. get_dones() aliases them
        # directly to the `died`/`time_out` tensors it also puts in terminated_dict/time_out_dict
        # (env._last_terminated = died, not a copy) -- those are the exact tensors env.step()
        # returns to the caller, and _reset_idx runs *before* that return. An in-place
        # `self._last_terminated[env_ids] = False` here used to zero out the done signal the
        # RL trainer was about to receive, for every env that just reset -- silently breaking
        # episode-boundary detection (GAE bootstrapping, and any extras["log"] consumer)
        # network-wide. No manual clearing is needed: get_dones() reassigns fresh tensors to
        # both buffers on the very next call regardless.

        for rob in self._robots:
            rob.reset(env_ids)

        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        desired_transposed = self._desired_pos_w.transpose(0, 1)
        initial_distances = torch.linalg.norm(desired_transposed - all_positions, dim=2)
        self._prev_distances = initial_distances
        self._prev_actions[env_ids] = 0.0
        self._rm_states[env_ids, :] = 0

        env_origins = self.scene.env_origins[env_ids]

        stage = self.curriculum_stage
        if stage == 1:
            set_stage1_positions(self, env_ids, env_origins)
        elif stage == 2:
            set_stage2_positions(self, env_ids, env_origins)
        elif stage == 3:
            set_stage3_positions(self, env_ids, env_origins)
        elif stage == 4:
            set_stage4_positions(self, env_ids, env_origins)
        elif stage == 5:
            set_stage5_positions(self, env_ids, env_origins)
        elif stage == 6:
            set_formation_positions(self, env_ids, env_origins)
        elif stage == 7:
            set_singlegoal_positions(self, env_ids, env_origins)
        elif stage in (8, 9):
            set_swarm_gravity_positions(self, env_ids, env_origins)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """Generate per-agent observations.

        Shared policy sees individual observations of dim determined by
        include_rm_in_obs (19 for baseline, 23 for fulltask).
        """
        ensure_cache_populated(self)

        all_lin_vels = torch.stack([rob.data.root_lin_vel_b for rob in self._robots], dim=0)
        all_ang_vels = torch.stack([rob.data.root_ang_vel_b for rob in self._robots], dim=0)
        all_gravities = torch.stack([rob.data.projected_gravity_b for rob in self._robots], dim=0)
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        all_quats = torch.stack([rob.data.root_quat_w for rob in self._robots], dim=0)

        desired_pos_w_transposed = self._desired_pos_w.transpose(0, 1)
        desired_pos_b, _ = subtract_frame_transforms(
            all_positions.reshape(-1, 3),
            all_quats.reshape(-1, 4),
            desired_pos_w_transposed.reshape(-1, 3),
        )
        desired_pos_b = desired_pos_b.reshape(self.num_drones, self.num_envs, 3)

        all_obs = _build_obs_tensor(
            self,
            all_lin_vels=all_lin_vels,
            all_ang_vels=all_ang_vels,
            all_gravities=all_gravities,
            desired_pos_b=desired_pos_b,
        )

        observations = {}
        for j in range(self.num_drones):
            observations[f"robot_{j}"] = all_obs[j]

        return observations

    def _get_states(self, dummy: bool = False) -> torch.Tensor:
        """Get centralized state for MAPPO critic (reuses cached computations).

        Returns concatenated observations from all agents.
        Shape: (num_envs, num_agents * obs_dim)
        """
        if dummy:
            return torch.zeros(self.num_envs, self.cfg.state_space, device=self.device)

        if not self._cache_valid:
            raise RuntimeError(
                "_get_states() called before cache is populated. "
                "This should never happen in normal workflow."
            )

        all_lin_vels = torch.stack([rob.data.root_lin_vel_b for rob in self._robots], dim=0)
        all_ang_vels = torch.stack([rob.data.root_ang_vel_b for rob in self._robots], dim=0)
        all_gravities = torch.stack([rob.data.projected_gravity_b for rob in self._robots], dim=0)
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        all_quats = torch.stack([rob.data.root_quat_w for rob in self._robots], dim=0)

        desired_pos_w_transposed = self._desired_pos_w.transpose(0, 1)
        desired_pos_b, _ = subtract_frame_transforms(
            all_positions.reshape(-1, 3),
            all_quats.reshape(-1, 4),
            desired_pos_w_transposed.reshape(-1, 3),
        )
        desired_pos_b = desired_pos_b.reshape(self.num_drones, self.num_envs, 3)

        all_obs = _build_obs_tensor(
            self,
            all_lin_vels=all_lin_vels,
            all_ang_vels=all_ang_vels,
            all_gravities=all_gravities,
            desired_pos_b=desired_pos_b,
        )

        all_obs = all_obs.transpose(0, 1)  # (num_envs, num_drones, obs_dim)
        state = all_obs.reshape(self.num_envs, -1)

        if self.cfg.include_distance_matrix_in_state:
            # Privileged info only the centralized critic sees: true world-frame
            # pairwise distance between every pair of agents (symmetric, zero diagonal)
            # -- meant for mappo_torchl.GraphAttentionCritic's attention bias, but just
            # extra flat input dims to any other critic.
            diff = all_positions.unsqueeze(1) - all_positions.unsqueeze(0)  # (D, D, E, 3)
            dist_matrix = diff.norm(dim=-1).permute(2, 0, 1)  # (E, D, D)
            state = torch.cat([state, dist_matrix.reshape(self.num_envs, -1)], dim=-1)

        return state

    # ------------------------------------------------------------------
    # Rewards and termination (delegated to module functions)
    # ------------------------------------------------------------------

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        if self.curriculum_stage == 6:
            if self.cfg.curriculum.stage6_simple_reward:
                return get_formation_rewards_simple(self)
            return get_formation_rewards(self)
        if self.curriculum_stage == 7:
            return get_formation_rewards_simple(self)
        if self.curriculum_stage in (8, 9):
            return get_swarm_gravity_rewards(self)
        return get_rewards(self)

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return get_dones(self)

    # ------------------------------------------------------------------
    # Debug visualization (delegated to module functions)
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        set_debug_vis_impl(self, debug_vis)

    def _debug_vis_callback(self, event):
        debug_vis_callback(self, event)


def _build_obs_tensor(
    env: BaseSwarmEnv,
    *,
    all_lin_vels: torch.Tensor,
    all_ang_vels: torch.Tensor,
    all_gravities: torch.Tensor,
    desired_pos_b: torch.Tensor,
) -> torch.Tensor:
    """Build observation tensor from pre-computed components.

    Shared by _get_observations() and _get_states().
    include_rm_in_obs controls whether RM state one-hot is appended.

    Returns:
        Tensor of shape (num_drones, num_envs, obs_dim)
    """
    rm_states_transposed = env._rm_states.transpose(0, 1)  # (num_drones, num_envs)
    rm_state_onehot = torch.nn.functional.one_hot(
        rm_states_transposed,
        num_classes=env.cfg.reward_cfg.num_rm_states,
    ).float()  # (num_drones, num_envs, 4)

    components = [
        all_lin_vels,                                # 3
        all_ang_vels,                                # 3
        all_gravities,                               # 3
        desired_pos_b,                                # 3
    ]
    if env.cfg.include_obstacle_in_obs:
        components += [
            env._cached_obstacle_dists.unsqueeze(-1),  # 1
            env._cached_obstacle_dir_b,                # 3
        ]
    components += [
        env._cached_neighbor_rel_pos_b,              # 3
        env._cached_neighbor_rel_vel_b,              # 3
        env._cached_mean_neighbor_pos_b,             # 3
        env._cached_mean_neighbor_vel_b,             # 3
    ]

    if env.cfg.include_rm_in_obs:
        components.append(rm_state_onehot)            # 4

    return torch.cat(components, dim=-1)


class FullTaskUAVSwarmEnv(BaseSwarmEnv):
    """FullTask variant: 23-dim observations including RM state one-hot."""

    cfg: FullTaskUAVSwarmEnvCfg


class BaselineUAVSwarmEnv(BaseSwarmEnv):
    """Baseline variant: 19-dim observations without RM state one-hot."""

    cfg: BaselineUAVSwarmEnvCfg


class FormationUAVSwarmEnv(BaseSwarmEnv):
    """Formation-assignment scalability task: scatter-spawn -> Hungarian-assigned
    V-formation slots, 28-dim observations without RM state one-hot."""

    cfg: FormationUAVSwarmEnvCfg


class SingleGoalUAVSwarmEnv(BaseSwarmEnv):
    """Single-UAV point-to-target diagnostic: same 28-dim obs/reward/termination path as
    Formation, independent per-agent goal instead of Hungarian-assigned V-formation slot."""

    cfg: SingleGoalUAVSwarmEnvCfg


class SwarmGravityUAVSwarmEnv(BaseSwarmEnv):
    """Swarm-gravity task: one shared target per env, agents self-organize around it via
    attraction + inter-agent repulsion, no fixed formation shape."""

    cfg: SwarmGravityUAVSwarmEnvCfg


class SwarmGravityV2UAVSwarmEnv(BaseSwarmEnv):
    """SwarmGravity variant for the critic-architecture experiment: same task, simplified
    24-dim obs (no obstacle fields), and an augmented 145-dim state (per-agent obs +
    pairwise distance matrix) for mappo_torchl.GraphAttentionCritic to consume."""

    cfg: SwarmGravityV2UAVSwarmEnvCfg
