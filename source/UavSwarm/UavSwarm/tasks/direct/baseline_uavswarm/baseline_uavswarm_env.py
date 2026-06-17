# ============================================================
# SwarmQuadEnv (Isaac Lab 2.3.0)
# Direct-style MARL: multiple Crazyflies per environment using Curriculum Learning
# Authors: Julio Rosa, adapted from CopyQuadEnv by NVIDIA Isaac Sim Team
# ============================================================
from __future__ import annotations

import torch
from isaaclab.utils.math import subtract_frame_transforms
from .baseline_uavswarm_env_cfg import BaselineUAVSwarmEnvCfg
from isaaclab.markers import SPHERE_MARKER_CFG  # isort: skip
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv
from .metrics_cfg import EpisodeMetrics
#import isaacsim.core.utils.prims as prim_utils
#from isaaclab.markers import VisualizationMarkers
#from isaaclab.scene import InteractiveSceneCfg



#from isaaclab.sensors import RayCaster
#check out https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html for more details on the functions implemented in DirectMARLEnv workflow
"""
Main idea of how use the Direct workflow when designing a task
###----- Overview of the Environment class structure:   
    _setup_scene(self): creates the environment, terrain, robots, etc. also  defines the distance between the parallel environments. This step is more the simulation scene configuration 
    _get_observations(self): gets the observations for each agent as a dict whe the obs should be in key {'policy':obs} when using a critic with different obs use {'policy':obs_policy,'critic':obs_critic}
    _pre_physics_step(self, actions): Mainly process the actions from the policy,performing computation like clipping or scaling, tranforming them to torques/forces, etc
    _apply_action(self): Apply the forces/torques to the robots, here is where the physics interaction happens
    _get_rewards(self): Compute the rewards for each agent, return as a dict with keys as agent names and values as reward tensors
    _get_dones(self): Compute the termination conditions for each agent, return as two dicts (terminated, time_out) with keys as agent names and values as boolean tensors
    _reset_idx(self, env_ids): Reset the environments specified by env_ids, resetting robot states, sampling new goals, etc

###-----Domain Randomization:
    Its also possible to implement domain randomization using the EventTerm and EventTermCfg
    Once the configclass for the randomization terms have been set up, the class must be added to the base config class for the task and be assigned to the variable events.
    # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["tiled_camera"] = self._tiled_camera
    @configclass
    class MyTaskConfig:
    events: EventCfg = EventCfg()

"""

class BaselineUAVSwarmEnv(DirectMARLEnv):
    """
    Direct-style MARL environment with N Crazyflies per env.
    - Actions: per-drone [thrust, mx, my, mz] ⇒ shape (num_agents, 4)
    - Observations: per-drone 23 dims ⇒ shape (num_agents, 23)
    - Rewards: per-drone terms + swarm cohesion/collision penalties
    """

    cfg: BaselineUAVSwarmEnvCfg

    def __init__(self, cfg: BaselineUAVSwarmEnvCfg, render_mode: str | None = None, **kwargs):
        
        self.num_drones = cfg.num_agents        
        self.global_step=0
        
        self._obstacles_built=False
        self.curriculum_stage = cfg.curriculum.active_stage
        
        cfg.episode_length_s = cfg.curriculum.get_episode_length()
        print(f"[INFO] Stage {cfg.curriculum.active_stage}: Episode length = {cfg.episode_length_s}s")
        # Initialize lists (before parent __init__)
        #self._num_agents_value = cfg.num_agents
        # ✅ CRITICAL: Initialize internal storage BEFORE parent init
        # self._num_agents_value = cfg.num_agents
        # self._agents_list = cfg.possible_agents.copy()
        # self._possible_agents_list = cfg.possible_agents.copy()
        

        self._robots = []
        self._body_ids = []
        self._obstacle_positions=None
        super().__init__(cfg, render_mode, **kwargs)
        print(f"[INFO] SuperClass var Episode length: {self.max_episode_length_s}s = {self.max_episode_length} steps")
        # Now device is available, initialize tensors
        self._actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, self.num_drones, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, self.num_drones, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, self.num_drones, 3, device=self.device)

        # Track termination reasons for logging
        self._last_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_timed_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ✅ NEW: Cache buffers for expensive computations
        # These are computed once in _get_observations() and reused in _get_rewards() and _get_states()
        self._cached_obstacle_dists = torch.zeros(
            self.num_drones, self.num_envs, device=self.device
        )  # (num_drones, num_envs)

        self._cached_neighbor_rel_pos_b = torch.zeros(
            self.num_drones, self.num_envs, 3, device=self.device
        )  # (num_drones, num_envs, 3)

        self._cached_neighbor_rel_vel_b = torch.zeros(
            self.num_drones, self.num_envs, 3, device=self.device
        )  # (num_drones, num_envs, 3)

        self._cached_obstacle_dir_b = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)
        self._cached_mean_neighbor_pos_b = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)
        self._cached_mean_neighbor_vel_b = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)

        self._prev_distances = torch.zeros(self.num_drones, self.num_envs, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)

        # ✅ NEW: Cache flag to ensure computations are done before use
        self._cache_valid = False


        # Logging
        # self._episode_sums = {
        #     key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        #     for key in ["lin_vel", "ang_vel", "distance_to_goal", "formation", "collision"]
        # }
        self._metrics = EpisodeMetrics.create(self.num_envs, self.device)
        print(f"[INFO] Initialized metrics tracker with {len(self._metrics.__dataclass_fields__)} metrics")

        # Get body indices and masses after robots are created
        self._body_ids = [rob.find_bodies("body")[0] for rob in self._robots]
        masses = [rob.root_physx_view.get_masses()[0].sum() for rob in self._robots]
        self._masses = torch.tensor(masses, device=self.device).view(1, self.num_drones)
        
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weights = (self._masses * self._gravity_magnitude).squeeze(0)  # (num_drones,)

        self.num_waypoints_per_agent = 3  # 3 obstacles → 3 waypoints (one behind each)
        
        #-----STAGE 3 Waypoint buffers
        # Waypoint paths: (num_envs, num_drones, num_waypoints, 3)
        self._waypoint_paths = torch.zeros(
            self.num_envs, self.num_drones, self.num_waypoints_per_agent, 3, 
            device=self.device
        )
        
        # Current waypoint index for each agent: (num_envs, num_drones)
        self._current_waypoint_idx = torch.zeros(
            self.num_envs, self.num_drones, dtype=torch.long, device=self.device
        )

        # -----------------------------------------------------
        # STAGE 5: Swarm waypoint buffers (shared across swarm)
        # -----------------------------------------------------
        self.num_swarm_waypoints = 3  # 3 waypoints through stacked X pattern
        
        # Swarm waypoint paths: (num_envs, num_waypoints, 3)
        # These are shared goals for the entire swarm (centroid targets)
        self._swarm_waypoint_paths = torch.zeros(
            self.num_envs, self.num_swarm_waypoints, 3,
            device=self.device
        )
        
        # Current swarm waypoint index: (num_envs,)
        self._current_swarm_waypoint_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Waypoint reached threshold (distance in meters)
        self._swarm_centroid = torch.zeros(self.num_envs, 3, device=self.device)
        self.waypoint_reach_threshold = 0.2
        self.swarm_waypoint_reach_threshold = 0.5  # Slightly larger for swarm centroid

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    

    ## MAIN ENVIRONMENT FUNCTIONS ##

    def _setup_scene(self):
        """Setup the scene with terrain and N robots per environment."""
        # Terrain
        #self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        #self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        #self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Create N Crazyflies per environment AND their sensors
        for i in range(self.num_drones):
            # Robot configuration
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
            
            #No more ray casters for now as they do not work properly with multiple obstacles and MARL envs
            
        # ✅ NEW: Build obstacles ONLY for active stage
        if self.curriculum_stage == 3:
            print("[INFO] Building Stage 3 obstacles at origin...")
            self._build_stage3_obstacles_at_origin()
        elif self.curriculum_stage == 5:
            print("[INFO] Building Stage 5 obstacles at origin...")
            self._build_stage5_obstacles_at_origin()
        else:
            print(f"[INFO] Stage {self.curriculum_stage} has no obstacles")
        

        # ✅ Clone environments (will replicate sensors automatically)
        self.scene.clone_environments(copy_from_source=False)
    
    
        # Filter collisions
        #if self.device == "cpu":
        #    self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        """Convert actions to thrust and moments for each drone.
        
        Args:
            actions: Dictionary mapping agent names to action tensors.
                     Each tensor has shape (num_envs, action_dim)
        """

        if self.curriculum_stage == 3:
            self._update_waypoint_goals()
        elif self.curriculum_stage == 5:
            self._update_swarm_waypoint_goals()

        # Compute swarm centroid for stages 4 and 5 (for visualization and waypoint logic)
        if self.curriculum_stage in [4, 5]:
            self._compute_swarm_centroid()

        # Convert dictionary to stacked tensor: (num_envs, num_drones, 4)
        actions_list = []
        for i in range(self.num_drones):
            agent_name = f"robot_{i}"
            actions_list.append(actions[agent_name])
        
        actions_tensor = torch.stack(actions_list, dim=1)  # (num_envs, num_drones, 4)
        self._actions = actions_tensor.clone().clamp(-1.0, 1.0)
        
        # Process each drone's actions
        for j in range(self.num_drones):
            thrust_cmd = (self._actions[:, j, 0] + 1.0) / 2.0  # [0, 1]
            self._thrust[:, j, 0, 2] = self.cfg.thrust_to_weight * self._robot_weights[j] * thrust_cmd
            self._moment[:, j, 0, :] = self.cfg.moment_scale * self._actions[:, j, 1:]

    def _apply_action(self):
        """Apply forces and torques to each robot."""
        for j, rob in enumerate(self._robots):
            rob.set_external_force_and_torque(
                self._thrust[:, j, :, :], 
                self._moment[:, j, :, :], 
                body_ids=self._body_ids[j]
            )
          # Next step will recompute in _get_observations()
        #self._cache_valid = False
        self._cache_valid = False  # Invalidate cache on action application
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments with curriculum-dependent goals & scene adjustments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

        # -----------------------------------------------------
        # 1. LOG EPISODIC METRICS
        # -----------------------------------------------------
        final_distances = []
        for j in range(self.num_drones):
            dist = torch.linalg.norm(
                self._desired_pos_w[env_ids, j, :]
                - self._robots[j].data.root_pos_w[env_ids],
                dim=1,
            )
            final_distances.append(dist)
        final_distance_to_goal = torch.stack(final_distances).mean()

        # ✅ USE .to_log_dict() METHOD
        log_dict = self._metrics.to_log_dict(
            env_ids=env_ids,
            max_episode_length=self.max_episode_length_s,
            prefix="Episode_Reward"
        )

        # ✅ ADD TERMINATION LOGS
        log_dict["Episode_Termination/died"] = torch.count_nonzero(
            self._last_terminated[env_ids]
        ).item()

        log_dict["Episode_Termination/time_out"] = torch.count_nonzero(
            self._last_timed_out[env_ids]
        ).item()

        # ✅ ADD DETAILED TERMINATION REASONS (if available)
        if hasattr(self, '_termination_reasons'):
            #print("episode termination reasons logged")
            log_dict["Episode_Termination/collision"] = torch.count_nonzero(
                self._termination_reasons['collision'][env_ids]
            ).item()
            
            log_dict["Episode_Termination/out_of_bounds"] = torch.count_nonzero(
                self._termination_reasons['out_of_bounds'][env_ids]
            ).item()
            
            log_dict["Episode_Termination/goal_reached"] = torch.count_nonzero(
                self._termination_reasons['goal_reached'][env_ids]
            ).item()

        # ✅ ADD FINAL METRICS
        log_dict["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        log_dict["Metrics/curriculum_stage"] = self.curriculum_stage

        # ✅ ADD CURRICULUM-SPECIFIC METRICS
        if self.curriculum_stage == 3:
            # Average waypoint progress
            avg_waypoint_progress = self._current_waypoint_idx[env_ids].float().mean().item()
            log_dict["Metrics/avg_waypoint_progress"] = avg_waypoint_progress / self.num_waypoints_per_agent
            
        elif self.curriculum_stage == 5:
            # Swarm waypoint progress
            avg_swarm_waypoint_progress = self._current_swarm_waypoint_idx[env_ids].float().mean().item()
            log_dict["Metrics/avg_swarm_waypoint_progress"] = avg_swarm_waypoint_progress / self.num_swarm_waypoints

        self.extras["log"] = log_dict

        # ✅ RESET METRICS USING .reset() METHOD
        self._metrics.reset(env_ids)
        self._cache_valid = False  # Invalidate cache on reset
        # Reset termination state
        self._last_terminated[env_ids] = False
        self._last_timed_out[env_ids] = False

        # -----------------------------------------------------
        # 2. RESET ROBOTS TO DEFAULT STATE
        # -----------------------------------------------------
        for rob in self._robots:
            rob.reset(env_ids)

        super()._reset_idx(env_ids)

        # desync episode starts
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

        # -----------------------------------------------------
        # 3. GET ENV ORIGINS
        # -----------------------------------------------------
        env_origins = self.scene.env_origins[env_ids]#self._terrain.env_origins[env_ids]

        # -----------------------------------------------------
        # 4. CURRICULUM LOGIC — SELECT THE RIGHT STAGE BEHAVIOR
        # -----------------------------------------------------
        # ✅ NEW: Only handle active stage (no offsets)
        stage = self.curriculum_stage
        #print(f"[INFO] Resetting to curriculum stage {stage}, Length: {self.cfg.curriculum.get_episode_length()}s")
        if stage == 1:
            self._set_stage1_positions(env_ids, env_origins)
        elif stage == 2:
            self._set_stage2_positions(env_ids, env_origins)
        elif stage == 3:
            self._set_stage3_positions(env_ids, env_origins)
        elif stage == 4:
            self._set_stage4_positions(env_ids, env_origins)
        elif stage == 5:
            self._set_stage5_positions(env_ids, env_origins)

    def _get_observations(self) -> dict:
        """Generate observations for shared policy MAPPO with centralized critic.
        
        Returns:
            Dictionary with two keys:
            - 'policy': Dict of per-agent observations for the actor {agent_name: obs}
            - 'critic': Dict of centralized states for the critic {agent_name: state}
            
            For shared policy:
            - Policy gets individual observations: (num_envs, obs_dim=23)
            - Critic gets concatenated observations from all agents: (num_envs, state_dim=num_agents*23)
        """
        
        # ✅ POPULATE CACHE IF NEEDED (after physics simulation)
        self._ensure_cache_populated()
        
        # Stack all robot data: (num_drones, num_envs, 3)
        all_lin_vels = torch.stack([rob.data.root_lin_vel_b for rob in self._robots], dim=0)
        all_ang_vels = torch.stack([rob.data.root_ang_vel_b for rob in self._robots], dim=0)
        all_gravities = torch.stack([rob.data.projected_gravity_b for rob in self._robots], dim=0)
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        all_quats = torch.stack([rob.data.root_quat_w for rob in self._robots], dim=0)
        
        # Transform desired positions to body frame (vectorized for all agents)
        desired_pos_w_transposed = self._desired_pos_w.transpose(0, 1)  # (num_drones, num_envs, 3)
        
        desired_pos_b, _ = subtract_frame_transforms(
            all_positions.reshape(-1, 3),
            all_quats.reshape(-1, 4),
            desired_pos_w_transposed.reshape(-1, 3)
        )
        desired_pos_b = desired_pos_b.reshape(self.num_drones, self.num_envs, 3)

        # Shape: (num_drones, num_envs, 28)
        all_obs = torch.cat([
            all_lin_vels,                                    # 3  [0:3]
            all_ang_vels,                                    # 3  [3:6]
            all_gravities,                                   # 3  [6:9]
            desired_pos_b,                                   # 3  [9:12]
            self._cached_obstacle_dists.unsqueeze(-1),       # 1  [12]
            self._cached_obstacle_dir_b,                     # 3  [13:16]
            self._cached_neighbor_rel_pos_b,                 # 3  [16:19]
            self._cached_neighbor_rel_vel_b,                 # 3  [19:22]
            self._cached_mean_neighbor_pos_b,                # 3  [22:25]
            self._cached_mean_neighbor_vel_b,                # 3  [25:28]
        ], dim=-1)  # (num_drones, num_envs, 28)
        
        # ✅ CREATE POLICY OBSERVATIONS (per-agent)
        observations = {}
        for j in range(self.num_drones):
            observations[f"robot_{j}"] = all_obs[j]  # (num_envs, 23)
        
        # ✅ RETURN SIMPLE DICTIONARY (no nested 'policy'/'critic' keys)
        # With separate: True, both policy and critic use the same observations
        return observations
        
    def _get_states(self, dummy: bool = False) -> torch.Tensor:
        """Get centralized state for MAPPO critic (reuses cached computations).
        
        Args:
            dummy: If True, returns a dummy state tensor with correct shape but invalid data.
                Used during initialization to infer state dimensions.
        
        Returns:
            Concatenated observations from all agents for centralized critic.
            Shape: (num_envs, num_agents * obs_dim)
        """
        # ✅ DUMMY MODE: Return dummy state for shape inference only
        if dummy:
            # Calculate expected state dimension
            #obs_dim = self.obs_spaces[self.possible_agents[0]].shape[0]  # e.g., 23
            #state_dim = self.num_agents * obs_dim  # e.g., 5 * 23 = 115
            
            # Return dummy tensor with correct shape
            dummy_state = torch.zeros(self.num_envs, self.cfg.state_space, device=self.device)
            return dummy_state
        
        # ✅ NORMAL MODE: Safety check and compute actual state
        # ✅ SAFETY CHECK: Ensure cache is valid
        if not self._cache_valid:
            raise RuntimeError(
                "_get_states() called before _get_observations()! "
                "Cache is invalid. This should never happen in normal workflow."
            )
        
        # ✅ REUSE ALREADY COMPUTED DATA
        # Stack all robot data: (num_drones, num_envs, 3)
        all_lin_vels = torch.stack([rob.data.root_lin_vel_b for rob in self._robots], dim=0)
        all_ang_vels = torch.stack([rob.data.root_ang_vel_b for rob in self._robots], dim=0)
        all_gravities = torch.stack([rob.data.projected_gravity_b for rob in self._robots], dim=0)
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        all_quats = torch.stack([rob.data.root_quat_w for rob in self._robots], dim=0)
        
        # Transform desired positions to body frame
        desired_pos_w_transposed = self._desired_pos_w.transpose(0, 1)
        
        desired_pos_b, _ = subtract_frame_transforms(
            all_positions.reshape(-1, 3),
            all_quats.reshape(-1, 4),
            desired_pos_w_transposed.reshape(-1, 3)
        )
        desired_pos_b = desired_pos_b.reshape(self.num_drones, self.num_envs, 3)

        all_obs = torch.cat([
            all_lin_vels,
            all_ang_vels,
            all_gravities,
            desired_pos_b,
            self._cached_obstacle_dists.unsqueeze(-1),
            self._cached_obstacle_dir_b,
            self._cached_neighbor_rel_pos_b,
            self._cached_neighbor_rel_vel_b,
            self._cached_mean_neighbor_pos_b,
            self._cached_mean_neighbor_vel_b,
        ], dim=-1)
        
        # Transpose to (num_envs, num_drones, 23)
        all_obs = all_obs.transpose(0, 1)
        
        # Flatten agents dimension: (num_envs, num_drones * 23)
        state = all_obs.reshape(self.num_envs, -1)
        
        return state

    def _base_reward(self) -> torch.Tensor:
        """Simple distance-based reward adapted from copy_quadenv.py for multi-agent use.
        
        Computes individual rewards for each agent using the proven simple formula:
        - Distance term: Main objective (exponentially decaying with tanh mapping)
        - Velocity penalties: Encourage smooth, stable flight
        
        Returns:
            Per-environment reward (mean across all agents), shape (num_envs,)
        """
        # Stack all robot data: (num_drones, num_envs, 3)
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        all_lin_vels = torch.stack([rob.data.root_lin_vel_b for rob in self._robots], dim=0)
        all_ang_vels = torch.stack([rob.data.root_ang_vel_b for rob in self._robots], dim=0)
        
        desired_transposed = self._desired_pos_w.transpose(0, 1)  # (num_drones, num_envs, 3)
        
        # ========================================
        # 1. DISTANCE TERM (per agent)
        # ========================================
        # Euclidean distance to goal
        distances = torch.linalg.norm(desired_transposed - all_positions, dim=2)  # (num_drones, num_envs)
        
        # Map distance to [0, 1] using tanh (same as copy_quadenv.py)
        distance_mapped = 1.0 - torch.tanh(distances / 0.8)  # (num_drones, num_envs)
        
        # Scale by reward weight
        distance_reward = distance_mapped * 15.0  # (num_drones, num_envs)
        
        # ========================================
        # 2. VELOCITY PENALTIES (per agent)
        # ========================================
        # Linear velocity penalty (encourage hovering when at goal)
        lin_vel_squared = torch.sum(all_lin_vels ** 2, dim=2)  # (num_drones, num_envs)
        lin_vel_penalty = lin_vel_squared * -0.05  # (num_drones, num_envs)
        
        # Angular velocity penalty (encourage stable orientation)
        ang_vel_squared = torch.sum(all_ang_vels ** 2, dim=2)  # (num_drones, num_envs)
        ang_vel_penalty = ang_vel_squared * -0.01  # (num_drones, num_envs)
        
        # ========================================
        # 3. COMBINE PER-AGENT REWARDS
        # ========================================
        # Sum components for each agent
        per_agent_reward = distance_reward + lin_vel_penalty + ang_vel_penalty  # (num_drones, num_envs)
        
        # ========================================
        # 4. AGGREGATE TO GLOBAL REWARD
        # ========================================
        # Mean across all agents to get environment-level reward
        global_reward = per_agent_reward.mean(dim=0)  # (num_envs,)
        
        # ========================================
        # 5. SAFETY PENALTIES (environment-level)
        # ========================================
        # Collision penalty (any agent out of bounds)
        agent_z = all_positions[:, :, 2]  # (num_drones, num_envs)
        too_low = agent_z < self.cfg.reward_cfg.min_flight_height
        too_high = agent_z > self.cfg.reward_cfg.max_flight_height
        collision = -(too_low | too_high).any(dim=0).float() * 10.0  # (num_envs,)
        
        # ========================================
        # 6. FINAL REWARD
        # ========================================
        reward = global_reward + collision  # (num_envs,)
        
        # ========================================
        # 7. LOGGING (OPTIONAL)
        # ========================================
        self._metrics.update(
            distance_to_goal=distances.mean(dim=0),  # Mean distance across agents
            lin_vel=lin_vel_penalty.abs().mean(dim=0),  # Mean linear velocity penalty
            ang_vel=ang_vel_penalty.abs().mean(dim=0),  # Mean angular velocity penalty
            collision=collision.abs(),
            mean_reward=reward,
        )
        
        return reward  # (num_envs,)



    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Purely additive reward — no RM state conditioning.

        r = pos_energy + progress + alignment
          - obstacle_cost   (stages 3, 5)
          - coop_cost       (stages 4, 5)
          - collision_penalty
          - jerk_penalty
        """
        # Reward constants
        K_POS    = 20.0   # inverse-quadratic position energy scale
        K_DELTA  = 3.0    # progress (distance delta) scale
        K_ALIGN  = 2.0    # velocity alignment scale
        K_OBS    = 3.0    # obstacle repulsion scale
        D_SAFE   = 1.5    # obstacle safety radius (m)
        D_INFL   = 3.0    # obstacle influence radius (m)
        K_COOP   = 2.0    # cooperation term scale
        D_OPT    = 1.75   # optimal neighbor distance (m)
        K_JERK   = 0.01   # jerk penalty scale
        K_COLL   = 10.0   # collision penalty

        self._ensure_cache_populated()

        # Robot state tensors: (num_drones, num_envs, 3)
        all_positions  = torch.stack([rob.data.root_pos_w     for rob in self._robots], dim=0)
        all_lin_vels   = torch.stack([rob.data.root_lin_vel_b for rob in self._robots], dim=0)
        all_lin_vels_w = torch.stack([rob.data.root_lin_vel_w for rob in self._robots], dim=0)

        desired = self._desired_pos_w.transpose(0, 1)  # (num_drones, num_envs, 3)

        # 1. Position energy — inverse quadratic potential
        distances     = torch.linalg.norm(desired - all_positions, dim=2)  # (num_drones, num_envs)
        pos_energy    = K_POS / (1.0 + distances ** 2)

        # 2. Progress — reward for reducing distance since last step
        distance_delta = self._prev_distances - distances
        self._prev_distances = distances.clone()
        progress = K_DELTA * distance_delta

        # 3. Velocity alignment — bonus for moving toward goal
        goal_dir      = desired - all_positions
        goal_dir_norm = goal_dir / (torch.linalg.norm(goal_dir, dim=2, keepdim=True) + 1e-8)
        vel_mag       = torch.linalg.norm(all_lin_vels_w, dim=2, keepdim=True) + 1e-8
        vel_dir_norm  = all_lin_vels_w / vel_mag
        cos_align     = torch.sum(goal_dir_norm * vel_dir_norm, dim=2)
        is_moving     = (vel_mag.squeeze(-1) > 0.1).float()
        alignment     = K_ALIGN * torch.clamp(cos_align, min=0.0) * is_moving

        # 4. Obstacle cost — additive repulsive penalty (stages 3, 5)
        if self.curriculum_stage in [3, 5]:
            influence     = torch.clamp(
                (D_INFL - self._cached_obstacle_dists) / (D_INFL - D_SAFE), 0.0, 1.0
            )
            obstacle_cost = K_OBS * influence ** 2
        else:
            obstacle_cost = torch.zeros_like(distances)

        # 5. Cooperation cost — penalty for deviation from optimal neighbor distance (stages 4, 5)
        if self.curriculum_stage in [4, 5]:
            diff          = all_positions.unsqueeze(1) - all_positions.unsqueeze(0)
            pairwise      = torch.linalg.norm(diff, dim=3)
            pairwise      = pairwise + torch.eye(self.num_drones, device=self.device).unsqueeze(2) * 1e6
            neighbor_dist = pairwise.min(dim=1)[0]
            coop_cost     = K_COOP * (neighbor_dist - D_OPT) ** 2
        else:
            coop_cost     = torch.zeros_like(distances)

        # 6. Per-drone additive reward
        per_drone = pos_energy + progress + alignment - obstacle_cost - coop_cost
        mean_reward = per_drone.mean(dim=0)  # (num_envs,)

        # 7. Collision penalty
        agent_z   = all_positions[:, :, 2]
        too_low   = agent_z < self.cfg.reward_cfg.min_flight_height
        too_high  = agent_z > self.cfg.reward_cfg.max_flight_height
        collision = -(too_low | too_high).any(dim=0).float() * K_COLL

        # 8. Jerk penalty
        action_diff  = self._actions - self._prev_actions
        jerk_penalty = -K_JERK * torch.sum(action_diff ** 2, dim=(1, 2))
        self._prev_actions = self._actions.clone()

        reward = mean_reward + collision + jerk_penalty
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-K_COLL)

        self._metrics.update(
            distance_to_goal=distances.mean(dim=0),
            collision=collision.abs(),
            mean_reward=reward,
        )

        return {f"robot_{i}": reward for i in range(self.num_drones)}

#----Termination Conditions with Curriculum Awareness----#  
    
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Check termination conditions with curriculum-aware logic.
        
        Termination reasons:
        1. Collision: Any drone too low (< 0.1m) or too high (> max_flight_height)
        2. Out of bounds: Any drone too far from environment origin
        3. Goal reached: All agents reached their goals (stage-dependent)
        4. Timeout: Episode exceeds max_episode_length
        
        Returns:
            Tuple of (terminated_dict, time_out_dict) where each is a dictionary
            mapping agent names to boolean tensors of shape (num_envs,)
        """
        # -----------------------------------------------------
        # 1. TIMEOUT TERMINATION (all stages)
        # -----------------------------------------------------
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # -----------------------------------------------------
        # 2. COLLISION TERMINATION (all stages)
        # -----------------------------------------------------
        # Died if any drone is too low or too high
        died_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for rob in self._robots:
            agent_z = rob.data.root_pos_w[:, 2]  # (num_envs,)
            
            # ✅ IMPROVED: Use configured height bounds
            too_low = agent_z < self.cfg.reward_cfg.min_flight_height  # Below ground/obstacles
            too_high = agent_z > self.cfg.reward_cfg.max_flight_height  # Above safe zone
            
            # Any agent collision causes environment termination
            died_collision = died_collision | too_low | too_high
        
        # -----------------------------------------------------
        # 3. OUT OF BOUNDS TERMINATION (all stages)
        # -----------------------------------------------------
        # Terminate if any drone strays too far from environment origin
        died_out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        env_origins = self.scene.env_origins#self.scene.env_origins  # (num_envs, 3)
        max_distance_from_origin = self.cfg.reward_cfg.max_distance_from_origin  # e.g., 50.0m
        
        for rob in self._robots:
            # Calculate XY distance from environment origin
            agent_pos_xy = rob.data.root_pos_w[:, :2]  # (num_envs, 2)
            origin_xy = env_origins[:, :2]  # (num_envs, 2)
            
            distance_from_origin = torch.linalg.norm(agent_pos_xy - origin_xy, dim=1)  # (num_envs,)
            
            # Terminate if any agent too far
            died_out_of_bounds = died_out_of_bounds | (distance_from_origin > max_distance_from_origin)
        
        # -----------------------------------------------------
        # 4. ✅ NEW: GOAL REACHED TERMINATION (curriculum-aware)
        # -----------------------------------------------------
        goal_reached = self._check_goal_reached()  # (num_envs,)
        
        # -----------------------------------------------------
        # 5. COMBINE TERMINATION CONDITIONS
        # -----------------------------------------------------
        # Died = collision OR out of bounds OR goal reached
        died = died_collision | died_out_of_bounds | goal_reached
        
        # -----------------------------------------------------
        # 6. LOGGING (for debugging and analysis)
        # -----------------------------------------------------
        # Store termination reasons for logging in _reset_idx
        self._last_terminated = died
        self._last_timed_out = time_out
        
        # ✅ NEW: Store detailed termination reasons
        if not hasattr(self, '_termination_reasons'):
            self._termination_reasons = {
                'collision': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                'out_of_bounds': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                'goal_reached': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            }
        
        self._termination_reasons['collision'] = died_collision
        self._termination_reasons['out_of_bounds'] = died_out_of_bounds
        self._termination_reasons['goal_reached'] = goal_reached
        
        # -----------------------------------------------------
        # 7. RETURN MARL-STYLE DICTIONARIES
        # -----------------------------------------------------
        # All agents share same termination conditions (cooperative setting)
        terminated_dict = {f"robot_{i}": died for i in range(self.num_drones)}
        time_out_dict = {f"robot_{i}": time_out for i in range(self.num_drones)}
        
        return terminated_dict, time_out_dict


    def _check_goal_reached(self) -> torch.Tensor:
        """Check if goals are reached based on current curriculum stage.
        
        Returns:
            Boolean tensor of shape (num_envs,) indicating which environments
            have completed their goals.
        """
        stage = self.curriculum_stage
        
        if stage == 1:
            # Stage 1: Individual hover - all agents within hover threshold
            return self._check_hover_goals_reached()
        
        elif stage == 2:
            # Stage 2: Individual point-to-point - all agents reach their goals
            return self._check_individual_goals_reached()
        
        elif stage == 3:
            # Stage 3: Individual waypoint navigation - all agents complete waypoint paths
            return self._check_waypoint_goals_reached()
        
        elif stage == 4:
            # Stage 4: Swarm navigation - all agents reach swarm goals
            return self._check_swarm_goals_reached()
        
        elif stage == 5:
            # Stage 5: Swarm waypoint navigation - swarm completes waypoint path
            return self._check_swarm_waypoint_goals_reached()
        
        else:
            # Unknown stage - no goal termination
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


    def _check_hover_goals_reached(self) -> torch.Tensor:
        """Check if all agents are hovering at their goal positions.
        
        Returns:
            Boolean tensor (num_envs,) - True if ALL agents within hover threshold
        """
        goal_reached_per_env = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        for j, rob in enumerate(self._robots):
            # Distance to goal
            distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[:, j, :] - rob.data.root_pos_w, 
                dim=1
            )  # (num_envs,)
            
            # Velocity magnitude (should be near zero for stable hover)
            velocity_mag = torch.linalg.norm(rob.data.root_lin_vel_w, dim=1)  # (num_envs,)
            
            # Agent reached goal if:
            # 1. Position within threshold (0.15m)
            # 2. Velocity below threshold (0.2 m/s)
            agent_reached = (distance_to_goal < self.cfg.reward_cfg.hover_position_threshold) & \
                        (velocity_mag < self.cfg.reward_cfg.hover_velocity_threshold)
            
            # ALL agents must reach goals
            goal_reached_per_env = goal_reached_per_env & agent_reached
        
        return goal_reached_per_env


    def _check_individual_goals_reached(self) -> torch.Tensor:
        """Check if all agents reached their individual point-to-point goals.
        
        Returns:
            Boolean tensor (num_envs,) - True if ALL agents within goal threshold
        """
        goal_reached_per_env = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        for j, rob in enumerate(self._robots):
            distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[:, j, :] - rob.data.root_pos_w, 
                dim=1
            )  # (num_envs,)
            
            # Agent reached goal if within threshold (0.3m for moving goals)
            agent_reached = distance_to_goal < self.cfg.reward_cfg.goal_position_threshold
            
            # ALL agents must reach goals
            goal_reached_per_env = goal_reached_per_env & agent_reached
        
        return goal_reached_per_env


    def _check_waypoint_goals_reached(self) -> torch.Tensor:
        """Check if all agents completed their waypoint paths (Stage 3).
        
        Returns:
            Boolean tensor (num_envs,) - True if ALL agents finished all waypoints
        """
        goal_reached_per_env = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        for j in range(self.num_drones):
            # Check if agent completed all waypoints
            # _current_waypoint_idx >= num_waypoints means completed
            completed = self._current_waypoint_idx[:, j] >= self.num_waypoints_per_agent
            
            # ALL agents must complete waypoints
            goal_reached_per_env = goal_reached_per_env & completed
        
        return goal_reached_per_env


    def _check_swarm_goals_reached(self) -> torch.Tensor:
        """Check if swarm reached goal formation (Stage 4).
        
        Returns:
            Boolean tensor (num_envs,) - True if swarm centroid within threshold
        """
        # Calculate swarm centroid
        swarm_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=1)  # (num_envs, num_drones, 3)
        swarm_centroid = swarm_positions.mean(dim=1)  # (num_envs, 3)
        
        # Calculate goal centroid (average of all agent goals)
        goal_centroid = self._desired_pos_w.mean(dim=1)  # (num_envs, 3)
        
        # Distance from swarm centroid to goal centroid
        centroid_distance = torch.linalg.norm(swarm_centroid - goal_centroid, dim=1)  # (num_envs,)
        
        # Swarm reached goal if centroid within threshold
        goal_reached = centroid_distance < self.cfg.reward_cfg.swarm_goal_threshold  # e.g., 0.5m
        
        return goal_reached


    def _check_swarm_waypoint_goals_reached(self) -> torch.Tensor:
        """Check if swarm completed waypoint path (Stage 5).
        
        Returns:
            Boolean tensor (num_envs,) - True if swarm finished all waypoints
        """
        # Check if swarm completed all waypoints
        # _current_swarm_waypoint_idx >= num_swarm_waypoints means completed
        completed = self._current_swarm_waypoint_idx >= self.num_swarm_waypoints
        
        return completed
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # Import at the top of the debug_vis block so it's available everywhere
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG
            
            # ---------------------------------------------------------
            # Goal position markers (yellow spheres)
            # ---------------------------------------------------------
            if not hasattr(self, "goal_pos_visualizers"):
                self.goal_pos_visualizers = []
                for i in range(self.num_drones):
                    marker_cfg = SPHERE_MARKER_CFG.copy()
                    marker_cfg.markers["sphere"].radius = 0.05
                    marker_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)  # green
                    marker_cfg.prim_path = f"/Visuals/Command/goal_position_{i}"
                    self.goal_pos_visualizers.append(VisualizationMarkers(marker_cfg))
            
            for viz in self.goal_pos_visualizers:
                viz.set_visibility(True)

            # ---------------------------------------------------------
            # Swarm centroid marker (blue arrow)
            # ---------------------------------------------------------
            if not hasattr(self, "centroid_visualizer"):
                centroid_marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                centroid_marker_cfg.prim_path = "/Visuals/Command/swarm_centroid"
                self.centroid_visualizer = VisualizationMarkers(centroid_marker_cfg)

            # Always show centroid (will be hidden in callback if not stages 4/5)
            if hasattr(self, "centroid_visualizer"):
                self.centroid_visualizer.set_visibility(True)

            # ---------------------------------------------------------
            # Stage 3: Individual waypoint markers (green cuboids)
            # ---------------------------------------------------------
            if not hasattr(self, "stage3_waypoint_visualizers"):
                self.stage3_waypoint_visualizers = []
                for i in range(self.num_drones):
                    for wp_idx in range(self.num_waypoints_per_agent):
                        marker_cfg = CUBOID_MARKER_CFG.copy()
                        marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                        marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)  # yellow-green
                        marker_cfg.prim_path = f"/Visuals/Command/stage3_waypoint_agent{i}_wp{wp_idx}"
                        self.stage3_waypoint_visualizers.append(VisualizationMarkers(marker_cfg))
            
            # Don't set visibility here - let _debug_vis_callback handle it based on current stage

            # ---------------------------------------------------------
            # Stage 5: Swarm waypoint markers (green cuboids)
            # ---------------------------------------------------------
            if not hasattr(self, "stage5_waypoint_visualizers"):
                self.stage5_waypoint_visualizers = []
                for wp_idx in range(self.num_swarm_waypoints):
                    marker_cfg = CUBOID_MARKER_CFG.copy()
                    marker_cfg.markers["cuboid"].size = (0.3, 0.3, 0.3)  # Larger for swarm waypoints
                    marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)  # green
                    marker_cfg.prim_path = f"/Visuals/Command/stage5_swarm_waypoint_{wp_idx}"
                    self.stage5_waypoint_visualizers.append(VisualizationMarkers(marker_cfg))
            
            # Don't set visibility here - let _debug_vis_callback handle it based on current stage

        # ------------------------------------------------------------------
        # Disable all visualizations
        # ------------------------------------------------------------------
        else:
            if hasattr(self, "goal_pos_visualizers"):
                for viz in self.goal_pos_visualizers:
                    viz.set_visibility(False)

            if hasattr(self, "centroid_visualizer"):
                self.centroid_visualizer.set_visibility(False)

            if hasattr(self, "stage3_waypoint_visualizers"):
                for viz in self.stage3_waypoint_visualizers:
                    viz.set_visibility(False)

            if hasattr(self, "stage5_waypoint_visualizers"):
                for viz in self.stage5_waypoint_visualizers:
                    viz.set_visibility(False)
                
    def _debug_vis_callback(self, event):
        """Update debug visualization markers (matching copy_quadenv.py)."""
        # Update goal position markers (always visible)
        if hasattr(self, "goal_pos_visualizers"):
            for i, viz in enumerate(self.goal_pos_visualizers):
                viz.visualize(self._desired_pos_w[:, i, :])
        
        # Update swarm centroid marker (only for stages 4 and 5)
        if hasattr(self, "centroid_visualizer"):
            if self.curriculum_stage in [4, 5]:
                self.centroid_visualizer.set_visibility(True)
                self.centroid_visualizer.visualize(self._swarm_centroid)
            else:
                self.centroid_visualizer.set_visibility(False)
        
        # Update stage 3 waypoint markers (green cuboids)
        if hasattr(self, "stage3_waypoint_visualizers"):
            if self.curriculum_stage == 3:
                # Show and update waypoints
                viz_idx = 0
                for agent_idx in range(self.num_drones):
                    for wp_idx in range(self.num_waypoints_per_agent):
                        # Get waypoint positions for all environments: (num_envs, 3)
                        waypoint_positions = self._waypoint_paths[:, agent_idx, wp_idx, :]
                        self.stage3_waypoint_visualizers[viz_idx].set_visibility(True)
                        self.stage3_waypoint_visualizers[viz_idx].visualize(waypoint_positions)
                        viz_idx += 1
            else:
                # Hide all stage 3 waypoints
                for viz in self.stage3_waypoint_visualizers:
                    viz.set_visibility(False)
        
        # Update stage 5 swarm waypoint markers (green cuboids)
        if hasattr(self, "stage5_waypoint_visualizers"):
            if self.curriculum_stage == 5:
                # Show and update waypoints
                for wp_idx in range(self.num_swarm_waypoints):
                    # Get swarm waypoint positions for all environments: (num_envs, 3)
                    waypoint_positions = self._swarm_waypoint_paths[:, wp_idx, :]
                    self.stage5_waypoint_visualizers[wp_idx].set_visibility(True)
                    self.stage5_waypoint_visualizers[wp_idx].visualize(waypoint_positions)
            else:
                # Hide all stage 5 waypoints
                for viz in self.stage5_waypoint_visualizers:
                    viz.set_visibility(False)

    def _update_waypoint_goals(self):
            """Vectorized waypoint update for all environments and agents."""
            all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
            all_positions = all_positions.transpose(0, 1)  # (num_envs, num_drones, 3)
            
            current_wp_idx = self._current_waypoint_idx  # (num_envs, num_drones)
            not_finished = current_wp_idx < self.num_waypoints_per_agent
            
            # Gather current waypoints
            env_idx = torch.arange(self.num_envs, device=self.device).view(-1, 1, 1)
            drone_idx = torch.arange(self.num_drones, device=self.device).view(1, -1, 1)
            wp_idx = current_wp_idx.unsqueeze(2).clamp(max=self.num_waypoints_per_agent - 1)
            coord_idx = torch.arange(3, device=self.device).view(1, 1, -1)
            
            current_waypoints = self._waypoint_paths[env_idx, drone_idx, wp_idx, coord_idx]
            distances = torch.linalg.norm(all_positions - current_waypoints, dim=2)
            
            # Check which agents reached waypoints
            reached = (distances < self.waypoint_reach_threshold) & not_finished
            
            # ✅ INCREMENT with clamping to max
            self._current_waypoint_idx = torch.where(
                reached,
                torch.clamp(current_wp_idx + 1, max=self.num_waypoints_per_agent),  # ✅ Explicit cap
                current_wp_idx
            )
            
            # Update goals
            next_wp_idx = self._current_waypoint_idx.unsqueeze(2).clamp(max=self.num_waypoints_per_agent - 1)
            next_waypoints = self._waypoint_paths[env_idx, drone_idx, next_wp_idx, coord_idx]
            
            self._desired_pos_w = torch.where(
                reached.unsqueeze(2).expand(-1, -1, 3),
                next_waypoints,
                self._desired_pos_w
            )

    def _update_swarm_waypoint_goals(self):
        """Update swarm goals based on centroid progress through waypoint path (Stage 5).
        
        When the swarm centroid reaches a waypoint, advance all agents to the next waypoint
        while maintaining their formation relative to the new target.
        """
        for env_idx in range(self.num_envs):
            # Get current waypoint index for this environment
            current_wp_idx = self._current_swarm_waypoint_idx[env_idx].item()
            
            # Check if swarm has completed all waypoints
            if current_wp_idx >= self.num_swarm_waypoints:
                continue  # Already at final waypoint
            
            # Calculate swarm centroid position
            swarm_positions = torch.stack([rob.data.root_pos_w[env_idx] for rob in self._robots], dim=0)  # (num_drones, 3)
            swarm_centroid = swarm_positions.mean(dim=0)  # (3,)
            
            # Get current waypoint target
            current_waypoint = self._swarm_waypoint_paths[env_idx, current_wp_idx]  # (3,)
            
            # Calculate distance from centroid to waypoint
            distance_to_waypoint = torch.linalg.norm(swarm_centroid - current_waypoint).item()
            
            # Check if waypoint is reached
            if distance_to_waypoint < self.swarm_waypoint_reach_threshold:
                # Advance to next waypoint
                next_wp_idx = current_wp_idx + 1
                
                if next_wp_idx < self.num_swarm_waypoints:
                    # Update to next waypoint
                    self._current_swarm_waypoint_idx[env_idx] = next_wp_idx
                    next_waypoint = self._swarm_waypoint_paths[env_idx, next_wp_idx]  # (3,)
                    
                    # Calculate formation offsets relative to current centroid
                    formation_offsets = swarm_positions - swarm_centroid.unsqueeze(0)  # (num_drones, 3)
                    
                    # Update each agent's goal to maintain formation around new waypoint
                    for j in range(self.num_drones):
                        goal_pos = next_waypoint + formation_offsets[j]
                        
                        self._desired_pos_w[env_idx, j, 0] = goal_pos[0]
                        self._desired_pos_w[env_idx, j, 1] = goal_pos[1]
                        self._desired_pos_w[env_idx, j, 2] = goal_pos[2]
                    
                    #print(f"[Stage 5] Env {env_idx}: Swarm reached waypoint {current_wp_idx}, advancing to {next_wp_idx}")
                else:
                    # Mark as completed
                    self._current_swarm_waypoint_idx[env_idx] = self.num_swarm_waypoints
                    #print(f"[Stage 5] Env {env_idx}: Swarm completed all waypoints!")

    
            
    def _ensure_cache_populated(self):
        """Populate cache if invalid (lazy evaluation).
        
        This is called by _get_rewards(), _get_observations(), and _get_states()
        to ensure cache is valid before use.
        """
        if self._cache_valid:
            return  # Already populated this step
        
        # Stack all robot data: (num_drones, num_envs, 3)
        all_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=0)
        all_quats = torch.stack([rob.data.root_quat_w for rob in self._robots], dim=0)
        
        self._cached_obstacle_dists, self._cached_obstacle_dir_b = \
            self._get_nearest_obstacle_distance_vectorized(all_positions, all_quats)

        (self._cached_neighbor_rel_pos_b,
         self._cached_neighbor_rel_vel_b,
         self._cached_mean_neighbor_pos_b,
         self._cached_mean_neighbor_vel_b) = \
            self._get_nearest_neighbor_data_vectorized(all_positions, all_quats)

        self._cache_valid = True

###------ Distance based helpers ------###
    
    def _get_nearest_neighbor_data_vectorized(
        self,
        all_positions: torch.Tensor,  # (num_drones, num_envs, 3)
        all_quats: torch.Tensor,       # (num_drones, num_envs, 4)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized nearest-neighbour + mean-pooled neighbour computation.

        Returns:
            nearest_pos_b:  (D, E, 3) — relative position of closest neighbour (body frame)
            nearest_vel_b:  (D, E, 3) — relative velocity of closest neighbour (body frame)
            mean_pos_b:     (D, E, 3) — mean relative position of ALL other drones (body frame)
            mean_vel_b:     (D, E, 3) — mean relative velocity of ALL other drones (body frame)
        """
        from isaaclab.utils.math import quat_apply_inverse

        if self.curriculum_stage in [1, 2, 3]:
            default_w = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)
            default_w[:, :, 0] = self.cfg.swarm_cfg.max_neighbor_distance
            default_b = quat_apply_inverse(
                all_quats.reshape(-1, 4), default_w.reshape(-1, 3)
            ).reshape(self.num_drones, self.num_envs, 3)
            zero_vel = torch.zeros_like(default_b)
            return default_b, zero_vel, default_b.clone(), zero_vel.clone()

        # diff[i, j, e, :] = pos_i - pos_j  shape: (D, D, E, 3)
        diff = all_positions.unsqueeze(1) - all_positions.unsqueeze(0)
        distances = torch.linalg.norm(diff, dim=3)  # (D, D, E)

        eye_mask = torch.eye(self.num_drones, device=self.device).unsqueeze(2)
        distances_masked = distances + eye_mask * 1e6
        nearest_idx = torch.argmin(distances_masked, dim=1)  # (D, E)

        env_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(0).expand(self.num_drones, -1)

        # --- Nearest neighbour ---
        nearest_rel_pos_w = all_positions[nearest_idx, env_idx] - all_positions  # (D, E, 3)
        rn = nearest_rel_pos_w.norm(dim=2, keepdim=True)
        nearest_rel_pos_w = torch.where(
            rn > self.cfg.swarm_cfg.max_neighbor_distance,
            nearest_rel_pos_w * (self.cfg.swarm_cfg.max_neighbor_distance / (rn + 1e-8)),
            nearest_rel_pos_w,
        )

        all_lin_vels_w = torch.stack([rob.data.root_lin_vel_w for rob in self._robots], dim=0)
        nearest_rel_vel_w = all_lin_vels_w[nearest_idx, env_idx] - all_lin_vels_w  # (D, E, 3)

        # --- Mean-pooled neighbours ---
        # -diff[i,j] = pos_j - pos_i; zero diagonal then mean over j
        eye_4d = torch.eye(self.num_drones, device=self.device).bool().unsqueeze(-1).unsqueeze(-1)
        mean_rel_pos_w = (-diff).masked_fill(eye_4d, 0.0).sum(dim=1) / (self.num_drones - 1)

        vel_diff = all_lin_vels_w.unsqueeze(0) - all_lin_vels_w.unsqueeze(1)  # (D, D, E, 3) [i,j]=vel_j-vel_i
        mean_rel_vel_w = vel_diff.masked_fill(eye_4d, 0.0).sum(dim=1) / (self.num_drones - 1)

        rm = mean_rel_pos_w.norm(dim=2, keepdim=True)
        mean_rel_pos_w = torch.where(
            rm > self.cfg.swarm_cfg.max_neighbor_distance,
            mean_rel_pos_w * (self.cfg.swarm_cfg.max_neighbor_distance / (rm + 1e-8)),
            mean_rel_pos_w,
        )

        # --- Body-frame transform (all four tensors) ---
        q_flat = all_quats.reshape(-1, 4)

        nearest_pos_b = quat_apply_inverse(q_flat, nearest_rel_pos_w.reshape(-1, 3)).reshape(
            self.num_drones, self.num_envs, 3
        )
        nearest_vel_b = quat_apply_inverse(q_flat, nearest_rel_vel_w.reshape(-1, 3)).reshape(
            self.num_drones, self.num_envs, 3
        )
        mean_pos_b = quat_apply_inverse(q_flat, mean_rel_pos_w.reshape(-1, 3)).reshape(
            self.num_drones, self.num_envs, 3
        )
        mean_vel_b = quat_apply_inverse(q_flat, mean_rel_vel_w.reshape(-1, 3)).reshape(
            self.num_drones, self.num_envs, 3
        )

        return nearest_pos_b, nearest_vel_b, mean_pos_b, mean_vel_b


    def _get_nearest_obstacle_distance_vectorized(
        self,
        all_positions: torch.Tensor,  # (num_drones, num_envs, 3)
        all_quats: torch.Tensor,       # (num_drones, num_envs, 4)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized obstacle distance + bearing direction for all agents.

        Returns:
            min_distances: (D, E)    — clamped distance to nearest obstacle
            obs_dir_b:     (D, E, 3) — unit vector toward nearest obstacle (body frame).
                                       Defaults to (1, 0, 0) when no obstacles present.
        """
        from isaaclab.utils.math import quat_apply_inverse

        max_dist = self.cfg.curriculum.max_obstacle_distance

        if self._obstacle_positions is None or self.curriculum_stage not in [3, 5]:
            min_distances = torch.full((self.num_drones, self.num_envs), max_dist, device=self.device)
            obs_dir_b = torch.zeros(self.num_drones, self.num_envs, 3, device=self.device)
            obs_dir_b[:, :, 0] = 1.0  # +x body-forward sentinel
            return min_distances, obs_dir_b

        # diff[d, e, k, :] = agent_pos[d,e] - obs_pos[k]  shape: (D, E, N_obs, 3)
        diff = all_positions.unsqueeze(2) - self._obstacle_positions.unsqueeze(0).unsqueeze(0)
        distances = diff.norm(dim=3)  # (D, E, N_obs)

        min_dist_values, nearest_obs_idx = distances.min(dim=2)  # (D, E) each
        min_distances = min_dist_values.clamp(0.0, max_dist)

        d_idx = torch.arange(self.num_drones, device=self.device).view(-1, 1).expand(-1, self.num_envs)
        e_idx = torch.arange(self.num_envs, device=self.device).view(1, -1).expand(self.num_drones, -1)
        nearest_diff = diff[d_idx, e_idx, nearest_obs_idx, :]  # (D, E, 3) = agent - obs
        obs_dir_w = -nearest_diff  # direction toward obstacle
        obs_dir_norm_w = obs_dir_w / (obs_dir_w.norm(dim=2, keepdim=True) + 1e-8)

        obs_dir_b = quat_apply_inverse(
            all_quats.reshape(-1, 4), obs_dir_norm_w.reshape(-1, 3)
        ).reshape(self.num_drones, self.num_envs, 3)

        return min_distances, obs_dir_b

###---------- Curriculum Methods ----------###
    

    # ✅ NEW: Obstacle builders at ORIGIN (no offsets)
    def _build_stage3_obstacles_at_origin(self):
        """Build stage 3 obstacles at environment origin (0, 0)."""
        from isaaclab.sim.spawners.shapes import CuboidCfg
        from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        
        source_env_idx = 0
        
        # ✅ GET SHARED PARAMETERS from config
        params = self.cfg.curriculum.get_stage3_params()
        obstacle_size = self.cfg.curriculum.obstacles_size
        base_height = obstacle_size[2] / 2.0  # Half of obstacle height
        
        global_wall_idx = 0
        
        for agent_idx in range(self.num_drones):
            lane_center_y = (agent_idx - (self.num_drones - 1) / 2.0) * params["lane_width"]
            
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
                
                wall_path = f"/World/envs/env_{source_env_idx}/obstacles/wall_{global_wall_idx}"
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
        
        # Collect obstacle positions
        self._collect_obstacle_positions_stage3()
        
        #print(f"[INFO] Built {global_wall_idx} Stage 3 obstacles using shared parameters")
    
    def _build_stage5_obstacles_at_origin(self):
        """Build stage 5 obstacles at environment origin (0, 0)."""
        from isaaclab.sim.spawners.shapes import CuboidCfg
        from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        
        source_env_idx = 0
        
        # ✅ REDUCED: Obstacle parameters
        x_offset = self.cfg.curriculum.stage5_obsx_offset  # 1.5m
        y_offset = self.cfg.curriculum.stage5_obsy_offset  # 2.0m
        obstacle_size = self.cfg.curriculum.obstacles_size  # ✅ Reduced from (1.2, 0.2, 15)
        #Inverted dimensions for vertical walls related to Y-axis travel
        obstacle_size = (obstacle_size[1], obstacle_size[0], obstacle_size[2])  # Reduced height
        base_height = obstacle_size[2] / 2.0  # Half of obstacle height
        
        dist_from_spawn_swarm = self.cfg.curriculum.dist_from_spawn_swarm  # 2.0m
        base_y = dist_from_spawn_swarm  # ✅ No Y offset
        base_x = 0.0  # ✅ At origin
        
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
        
        stage5_start_idx = self.num_drones * 3
        
        for local_idx, (wall_x, wall_y, wall_z) in enumerate(wall_positions):
            global_wall_idx = stage5_start_idx + local_idx
            
            wall_path = f"/World/envs/env_{source_env_idx}/obstacles/wall_{global_wall_idx}"
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
        
        # Collect obstacle positions
        self._collect_obstacle_positions_stage5()

    def _collect_obstacle_positions_stage3(self):
        """Collect Stage 3 obstacle positions using shared parameters."""
        obstacle_positions = []
        
        # ✅ GET SHARED PARAMETERS from config
        params = self.cfg.curriculum.get_stage3_params()
        obstacle_size = self.cfg.curriculum.obstacles_size
        base_height = obstacle_size[2] / 2.0
        
        for agent_idx in range(self.num_drones):
            lane_center_y = (agent_idx - (self.num_drones - 1) / 2.0) * params["lane_width"]
            
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
        
        self._obstacle_positions = torch.tensor(
            obstacle_positions, 
            dtype=torch.float32, 
            device=self.device
        )
        
        print(f"[INFO] Collected {len(obstacle_positions)} Stage 3 obstacle positions")

    def _collect_obstacle_positions_stage5(self):
        """Collect Stage 5 obstacle positions matching REDUCED parameters."""
        obstacle_positions = []
        
        # ✅ MATCH _build_stage5_obstacles_at_origin() parameters
        x_offset = self.cfg.curriculum.stage5_obsx_offset  # 1.5m
        y_offset = self.cfg.curriculum.stage5_obsy_offset  # 2.0m
        base_height = 4.0  # ✅ REDUCED from 2.5
        dist_from_spawn_swarm = self.cfg.curriculum.dist_from_spawn_swarm
        base_y = dist_from_spawn_swarm  # ✅ No Y offset
        base_x = 0.0  # ✅ At origin
        
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
        
        # Convert to tensor and store
        self._obstacle_positions = torch.tensor(
            obstacle_positions, 
            dtype=torch.float32, 
            device=self.device
        )
        
        #print(f"[INFO] Collected {len(obstacle_positions)} Stage 5 obstacle positions")

    def _set_stage1_positions(self, env_ids, env_origins):
        """Set hover goals - simplified grid version."""
        ##This function places the drones in a grid formation at the start of the episode and assigns each drone a goal position directly above its start position at a certain height.
        num_reset_envs = len(env_ids)
        
        # Create grid of start positions
        grid_size = int(torch.ceil(torch.sqrt(torch.tensor(self.num_drones, dtype=torch.float32))))
        spacing = torch.zeros(1,device=self.device).uniform_(self.cfg.curriculum.spawn_grid_spacing_range[0], self.cfg.curriculum.spawn_grid_spacing_range[1])#max(0.5, self.cfg.min_safe_distance)
        
        for env_idx in range(num_reset_envs):
            # Random permutation for this environment
            perm = torch.randperm(self.num_drones, device=self.device)
            
            # Sample heights
            start_heights = torch.zeros(self.num_drones, device=self.device).uniform_(0.6, 1.0)
            min_height = self.cfg.curriculum.goal_height_range[0]
            max_height = self.cfg.curriculum.goal_height_range[1]
            goal_heights = torch.zeros(self.num_drones, device=self.device).uniform_(min_height, max_height) 
            
            for j, rob in enumerate(self._robots):
                env_id_single = env_ids[env_idx].unsqueeze(0)
                
                # Grid position
                grid_idx = perm[j].item()
                grid_x = (grid_idx % grid_size) * spacing - (grid_size * spacing / 2.0)
                grid_y = (grid_idx // grid_size) * spacing - (grid_size * spacing / 2.0)
                
                # Reset robot
                joint_pos = rob.data.default_joint_pos[env_id_single]
                joint_vel = rob.data.default_joint_vel[env_id_single]
                default_root_state = rob.data.default_root_state[env_id_single].clone()
                
                # Start position
                default_root_state[:, 0] = env_origins[env_idx, 0] + grid_x
                default_root_state[:, 1] = env_origins[env_idx, 1] + grid_y
                default_root_state[:, 2] = start_heights[j]
                
                    # Start position
                start_x = env_origins[env_idx, 0] + grid_x
                start_y = env_origins[env_idx, 1] + grid_y
                start_z = start_heights[j]

                # ✅ DEBUG: Print drone positions
                # if env_idx == 0:  # Only print for first environment
                #     print(f"  - Drone {j}: pos=({start_x.item():.2f}, {start_y.item():.2f}, {start_z.item():.2f}), "
                #         f"goal=({start_x.item():.2f}, {start_y.item():.2f}, {goal_heights[j].item():.2f})")
                


                rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
                rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
                rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)
                
                # Goal position (minimal XY drift)
                xy_noise = torch.zeros(2, device=self.device).uniform_(-0.05, 0.05)
                self._desired_pos_w[env_id_single, j, 0] = default_root_state[0, 0] + xy_noise[0]
                self._desired_pos_w[env_id_single, j, 1] = default_root_state[0, 1] + xy_noise[1]
                self._desired_pos_w[env_id_single, j, 2] = goal_heights[j]
    
    def _set_stage2_positions(self, env_ids, env_origins):
        """Set individual point-to-point goals for curriculum stage 2.
        
        Agents must learn to:
        1. Navigate stably in XY plane at different heights
        2. Rotate (yaw) to face the goal direction
        3. Reach distant goals while avoiding inter-agent collisions
        
        Args:
            env_ids: Indices of environments to reset
            env_origins: Origins of environments, shape (num_reset_envs, 3)
        """
        num_reset_envs = len(env_ids)
        
        # Create grid of start positions
        grid_size = int(torch.ceil(torch.sqrt(torch.tensor(self.num_drones, dtype=torch.float32))))
        spacing = torch.zeros(1,device=self.device).uniform_(self.cfg.curriculum.spawn_grid_spacing_range[0], self.cfg.curriculum.spawn_grid_spacing_range[1])
        
        # ✅ Use configured height range
        min_height = self.cfg.curriculum.goal_height_range[0]
        max_height = self.cfg.curriculum.goal_height_range[1]
        
        # Calculate height stratification to prevent collisions
        # Each drone operates on a different Z-plane
        z_spacing = self.cfg.curriculum.stage2_zdist_xy_plane
        base_height = min_height  # Start from minimum height
        
        for env_idx in range(num_reset_envs):
            # Random permutation for grid assignment
            perm = torch.randperm(self.num_drones, device=self.device)
            
            # Assign heights to create vertical separation
            # Heights increase with drone index to create layered formation
            heights = torch.arange(self.num_drones, device=self.device, dtype=torch.float32)
            heights = base_height + heights * z_spacing
            
            # ✅ Clamp heights to stay within configured range
            heights = torch.clamp(heights, min=min_height, max=max_height)
            
            # Randomize height assignment (shuffle which drone gets which height layer)
            height_perm = torch.randperm(self.num_drones, device=self.device)
            assigned_heights = heights[height_perm]
            
            for j, rob in enumerate(self._robots):
                env_id_single = env_ids[env_idx].unsqueeze(0)
                
                # -----------------------------------------------------
                # 1. START POSITION: Grid formation with unique heights
                # -----------------------------------------------------
                grid_idx = perm[j].item()
                grid_x = (grid_idx % grid_size) * spacing - (grid_size * spacing / 2.0)
                grid_y = (grid_idx // grid_size) * spacing - (grid_size * spacing / 2.0)
                
                # Reset robot
                joint_pos = rob.data.default_joint_pos[env_id_single]
                joint_vel = rob.data.default_joint_vel[env_id_single]
                default_root_state = rob.data.default_root_state[env_id_single].clone()
                
                # Set start position
                start_x = env_origins[env_idx, 0] + grid_x
                start_y = env_origins[env_idx, 1] + grid_y
                start_z = assigned_heights[j].item()
                
                default_root_state[:, 0] = start_x
                default_root_state[:, 1] = start_y
                default_root_state[:, 2] = start_z
                
                # Write to simulation
                rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
                rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
                rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)
                
                # -----------------------------------------------------
                # 2. GOAL POSITION: Distant in XY, similar Z
                # -----------------------------------------------------
                # Sample distance and angle for goal
                goal_distance = torch.zeros(1, device=self.device).uniform_(
                    2.0, 
                    self.cfg.curriculum.stage2_goal_distance
                ).item()
                
                # Random angle in [0, 2π] to encourage yaw rotation
                # This ensures the drone must turn to face the goal
                goal_angle = torch.zeros(1, device=self.device).uniform_(
                    0.0, 
                    2.0 * torch.pi
                ).item()
                
                # Calculate goal XY offset from start position
                goal_offset_x = goal_distance * torch.cos(torch.tensor(goal_angle, device=self.device))
                goal_offset_y = goal_distance * torch.sin(torch.tensor(goal_angle, device=self.device))
                
                # Goal position: start + offset
                goal_x = start_x + goal_offset_x
                goal_y = start_y + goal_offset_y
                
                # ✅ FIXED: Goal height with variation, clamped to configured range
                # Goal Z stays close to start Z with small variation
                goal_z_noise = torch.zeros(1, device=self.device).uniform_(-0.3, 0.3).item()
                goal_z = torch.clamp(
                    torch.tensor(start_z + goal_z_noise, device=self.device),
                    min=min_height,  # Never below minimum safe height
                    max=max_height   # Never above maximum height
                ).item()
                
                # Set goal
                self._desired_pos_w[env_id_single, j, 0] = goal_x
                self._desired_pos_w[env_id_single, j, 1] = goal_y
                self._desired_pos_w[env_id_single, j, 2] = goal_z
 
    def _set_stage3_positions(self, env_ids, env_origins):
        """Set individual obstacle course navigation with waypoint-based goals."""
        num_reset_envs = len(env_ids)
        offset_x, offset_y = (0, 0)
        
        # ✅ GET SHARED PARAMETERS from config
        params = self.cfg.curriculum.get_stage3_params()
        
        # Height range
        min_height = self.cfg.curriculum.goal_height_range[0]
        max_height = self.cfg.curriculum.goal_height_range[1]
        
        # Agent spawn parameters
        spawn_x = -torch.zeros(num_reset_envs, device=self.device).uniform_(
            0.5,
            1.0
        )  # Spawn slightly behind course start
        
        for env_idx in range(num_reset_envs):
            env_id_int = env_ids[env_idx].item()
            env_id_single = env_ids[env_idx].unsqueeze(0)
            
            perm = torch.randperm(self.num_drones, device=self.device)
            base_height = torch.zeros(1, device=self.device).uniform_(min_height, max_height).item()
            
            for j, rob in enumerate(self._robots):
                # -----------------------------------------------------
                # AGENT START POSITION
                # -----------------------------------------------------
                agent_lane = perm[j].item()
                lane_center_y = offset_y + (agent_lane - (self.num_drones - 1) / 2.0) * params["lane_width"]
                
                start_x = env_origins[env_idx, 0] + spawn_x[env_idx]
                start_y = env_origins[env_idx, 1] + lane_center_y
                start_z = base_height
                
                # Reset robot
                joint_pos = rob.data.default_joint_pos[env_id_single]
                joint_vel = rob.data.default_joint_vel[env_id_single]
                default_root_state = rob.data.default_root_state[env_id_single].clone()
                
                default_root_state[:, 0] = start_x
                default_root_state[:, 1] = start_y
                default_root_state[:, 2] = start_z
                
                rob.write_root_pose_to_sim(default_root_state[:, :7], env_id_single)
                rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_id_single)
                rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id_single)
                
                # -----------------------------------------------------
                # GENERATE WAYPOINT PATH
                # -----------------------------------------------------
                for wp_idx in range(3):
                    # Obstacle center X position
                    obs_x = env_origins[env_idx, 0] + offset_x + params["course_start_x"] + wp_idx * params["obstacle_spacing_x"]
                    
                    # Zig-zag Y pattern
                    if wp_idx == 0:
                        obs_y = env_origins[env_idx, 1] + lane_center_y
                    elif wp_idx == 1:
                        obs_y = env_origins[env_idx, 1] + lane_center_y - params["lateral_offset"]
                    else:
                        obs_y = env_origins[env_idx, 1] + lane_center_y + params["lateral_offset"]
                    
                    # Waypoint BEHIND obstacle
                    waypoint_x = obs_x + params["waypoint_distance_behind"]
                    waypoint_y = obs_y
                    
                    z_variation = torch.zeros(1, device=self.device).uniform_(-0.2, 0.2).item()
                    waypoint_z = torch.clamp(
                        torch.tensor(base_height + z_variation, device=self.device),
                        min=min_height,
                        max=max_height
                    ).item()
                    
                    # Store waypoint
                    self._waypoint_paths[env_id_int, j, wp_idx, 0] = waypoint_x
                    self._waypoint_paths[env_id_int, j, wp_idx, 1] = waypoint_y
                    self._waypoint_paths[env_id_int, j, wp_idx, 2] = waypoint_z
                
                # Reset waypoint index
                self._current_waypoint_idx[env_id_int, j] = 0
                
                # Set initial goal
                self._desired_pos_w[env_id_single, j, :] = self._waypoint_paths[env_id_int, j, 0, :]
        
        #print(f"[INFO] Stage 3 reset complete using shared parameters")
       
    def _set_stage4_positions(self, env_ids, env_origins):
        """Set swarm navigation goals with formation aligned to travel direction."""
        num_reset_envs = len(env_ids)
        
        spawn_heights = torch.zeros(num_reset_envs, device=self.device).uniform_(
            self.cfg.curriculum.goal_height_range[0], 
            self.cfg.curriculum.goal_height_range[1]
        )
        #min_goal_height = self.cfg.curriculum.goal_height_range[0]
        #max_goal_height = self.cfg.curriculum.goal_height_range[1]
        # Start positions: Inverted V formation
        formation_positions = self.get_inverted_v_formation(env_ids, env_origins, spawn_heights)
        
        # Reset robots
        for j, rob in enumerate(self._robots):
            joint_pos = rob.data.default_joint_pos[env_ids]
            joint_vel = rob.data.default_joint_vel[env_ids]
            default_root_state = rob.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] = formation_positions[:, j, :]
            
            rob.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Goal positions
        for env_idx in range(num_reset_envs):
            env_id_single = env_ids[env_idx]
            
            # Sample translation
            translation_distance = torch.zeros(1, device=self.device).uniform_(
                self.cfg.curriculum.swarm_translation_distance_range[0],
                self.cfg.curriculum.swarm_translation_distance_range[1]
            ).item()
            
            translation_angle = torch.zeros(1, device=self.device).uniform_(0.0, 2.0 * torch.pi).item()
            
            translation_x = translation_distance * torch.cos(torch.tensor(translation_angle, device=self.device))
            translation_y = translation_distance * torch.sin(torch.tensor(translation_angle, device=self.device))
            
            # ALIGN V-formation to point toward travel direction
            # rotation_angle = translation_angle (V apex points toward goal)
            rotation_angle = translation_angle
            
            cos_theta = torch.cos(torch.tensor(rotation_angle, device=self.device))
            sin_theta = torch.sin(torch.tensor(rotation_angle, device=self.device))
            
            formation_center = formation_positions[env_idx].mean(dim=0)
            swarm_goal_height = spawn_heights[env_idx] + torch.zeros(1, device=self.device).uniform_(-0.5, 0.5).item()
            for j in range(self.num_drones):
                start_pos = formation_positions[env_idx, j]
                relative_pos = start_pos[:2] - formation_center[:2]
                
                # Rotate relative position
                rotated_x = cos_theta * relative_pos[0] - sin_theta * relative_pos[1]
                rotated_y = sin_theta * relative_pos[0] + cos_theta * relative_pos[1]
                
                # Translate to goal
                goal_x = formation_center[0] + translation_x + rotated_x
                goal_y = formation_center[1] + translation_y + rotated_y
                goal_z = start_pos[2] + swarm_goal_height
                
                self._desired_pos_w[env_id_single, j, 0] = goal_x
                self._desired_pos_w[env_id_single, j, 1] = goal_y
                self._desired_pos_w[env_id_single, j, 2] = goal_z

    def _set_stage5_positions(self, env_ids, env_origins):
        """Set swarm waypoint navigation through stacked X obstacle pattern.
        
        Swarm navigates through 3 waypoints placed at the centers of the X pattern gaps:
        - Waypoint 1: Center of bottom X (between wall1, wall2, wall3)
        - Waypoint 2: Center of middle gap (between wall4, wall5, wall6)
        - Waypoint 3: Center of top X (between wall6, wall7, wall8)
        
        The formation is rotated 90° to face the +Y direction (toward obstacles).
        
        Args:
            env_ids: Indices of environments to reset
            env_origins: Origins of environments, shape (num_reset_envs, 3)
        """
        num_reset_envs = len(env_ids)
        offset_x, offset_y = (0, 0)
        
        # Get obstacle configuration
        x_offset = self.cfg.curriculum.stage5_obsx_offset
        y_offset = self.cfg.curriculum.stage5_obsy_offset
        
        # Sample spawn heights
        spawn_heights = torch.zeros(num_reset_envs, device=self.device).uniform_(
            self.cfg.curriculum.goal_height_range[0], 
            self.cfg.curriculum.goal_height_range[1]
        )
        
        # ✅ NEW: Sample different Y-distance from spawn for each environment
        dist_y_from_spawn_swarm = torch.zeros(num_reset_envs, device=self.device).uniform_(
            0.8,  # Minimum distance
            1.5   # Maximum distance
        )
        
        # -----------------------------------------------------
        # 1. START POSITIONS: Inverted V formation ROTATED to face +Y
        # -----------------------------------------------------
        offset_origins = env_origins.clone()
        offset_origins[:, 0] += offset_x
        offset_origins[:, 1] += offset_y
        
        # Get default formation (apex points in -X direction)
        formation_positions = self.get_inverted_v_formation(env_ids, offset_origins, spawn_heights)
        
        # Rotate formation 90° counterclockwise to face +Y direction
        # Rotation matrix for 90° CCW: [0, -1; 1, 0]
        for env_idx in range(num_reset_envs):
            formation_center = formation_positions[env_idx].mean(dim=0)  # (3,)
            
            for j in range(self.num_drones):
                # Get relative position from formation center
                relative_pos = formation_positions[env_idx, j, :2] - formation_center[:2]  # (2,)
                
                # Apply 90° CCW rotation
                rotated_x = -relative_pos[1]  # New X = -Old Y
                rotated_y = -relative_pos[0]   # New Y = Old X
                
                # Update position
                formation_positions[env_idx, j, 0] = formation_center[0] + rotated_x
                formation_positions[env_idx, j, 1] = formation_center[1] + rotated_y
        
        # Reset robots with rotated formation positions
        for j, rob in enumerate(self._robots):
            joint_pos = rob.data.default_joint_pos[env_ids]
            joint_vel = rob.data.default_joint_vel[env_ids]
            default_root_state = rob.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] = formation_positions[:, j, :]
            
            rob.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            rob.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            rob.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # -----------------------------------------------------
        # 2. SWARM WAYPOINT PATHS (3 waypoints through X pattern)
        # -----------------------------------------------------
        min_goal_height = self.cfg.curriculum.goal_height_range[0]
        max_goal_height = self.cfg.curriculum.goal_height_range[1]
        
        # ✅ UPDATED: Use per-environment random distance
        for env_idx in range(num_reset_envs):
            env_id_int = env_ids[env_idx].item()
            
            # ✅ Get this environment's specific Y-distance
            env_dist_y = dist_y_from_spawn_swarm[env_idx].item()
            
            base_height = spawn_heights[env_idx].item()
            swarm_goal_height = torch.zeros(1, device=self.device).uniform_(-0.5, 0.5).item()
            
            # ✅ Waypoint 1: Gap in bottom X (use randomized distance)
            wp1_x = env_origins[env_idx, 0] + offset_x
            wp1_y = env_origins[env_idx, 1] + offset_y + env_dist_y + 0.75 * y_offset  # ✅ Per-env distance
            wp1_z = base_height + swarm_goal_height
            
            self._swarm_waypoint_paths[env_id_int, 0, 0] = wp1_x
            self._swarm_waypoint_paths[env_id_int, 0, 1] = wp1_y
            self._swarm_waypoint_paths[env_id_int, 0, 2] = wp1_z
            
            # ✅ Waypoint 2: Gap in middle (use randomized distance)
            wp2_x = env_origins[env_idx, 0] + offset_x
            wp2_y = env_origins[env_idx, 1] + offset_y + env_dist_y + 1.25 * y_offset  # ✅ Per-env distance
            wp2_z = base_height + swarm_goal_height
            
            self._swarm_waypoint_paths[env_id_int, 1, 0] = wp2_x
            self._swarm_waypoint_paths[env_id_int, 1, 1] = wp2_y
            self._swarm_waypoint_paths[env_id_int, 1, 2] = wp2_z
            
            # ✅ Waypoint 3: Final position beyond top X (use randomized distance)
            wp3_x = env_origins[env_idx, 0] + offset_x
            wp3_y = env_origins[env_idx, 1] + offset_y + env_dist_y + 2.5 * y_offset  # ✅ Per-env distance
            wp3_z = base_height + swarm_goal_height
            
            self._swarm_waypoint_paths[env_id_int, 2, 0] = wp3_x
            self._swarm_waypoint_paths[env_id_int, 2, 1] = wp3_y
            self._swarm_waypoint_paths[env_id_int, 2, 2] = wp3_z
            
            # Reset current waypoint index
            self._current_swarm_waypoint_idx[env_id_int] = 0
        
        # -----------------------------------------------------
        # 3. SET INITIAL GOALS (formation around first waypoint)
        # -----------------------------------------------------
        for env_idx in range(num_reset_envs):
            env_id_single = env_ids[env_idx]
            env_id_int = env_id_single.item()
            
            # Get first waypoint (swarm centroid target)
            swarm_target = self._swarm_waypoint_paths[env_id_int, 0, :]  # (3,)
            
            # Calculate formation center (after rotation)
            formation_center = formation_positions[env_idx].mean(dim=0)  # (3,)
            
            # Set individual goals to maintain formation relative to swarm target
            for j in range(self.num_drones):
                start_pos = formation_positions[env_idx, j]
                relative_pos = start_pos - formation_center
                
                # Goal = swarm_target + formation_offset
                goal_pos = swarm_target + relative_pos
                
                self._desired_pos_w[env_id_single, j, 0] = goal_pos[0]
                self._desired_pos_w[env_id_single, j, 1] = goal_pos[1]
                self._desired_pos_w[env_id_single, j, 2] = goal_pos[2]

###---- Swarm formation methods ----###
    def _compute_swarm_centroid(self):
            """Compute swarm centroid position for each environment (Stages 4 & 5).
            
            Calculates the mean position of all drones in each environment.
            Updates the _swarm_centroid buffer: (num_envs, 3)
            """
            # Stack all drone positions: (num_envs, num_drones, 3)
            swarm_positions = torch.stack([rob.data.root_pos_w for rob in self._robots], dim=1)
            
            # Compute centroid as mean across all drones: (num_envs, 3)
            self._swarm_centroid = swarm_positions.mean(dim=1)

    def get_inverted_v_formation(self, env_ids: torch.Tensor, env_origins: torch.Tensor, spawn_heights: torch.Tensor) -> torch.Tensor:
        """Calculate inverted V formation positions for all drones in specified environments.
        
        Args:
            env_ids: Indices of environments to reset
            env_origins: Origins of the environments being reset, shape (num_reset_envs, 3)
            spawn_heights: Height offset for each environment, shape (num_reset_envs,)
        
        Returns:
            Tensor of shape (num_reset_envs, num_drones, 3) with absolute world positions 
            for each drone in each environment.
        """
        num_reset_envs = len(env_ids)
        
        # Inverted V: apex at front (negative X), wings spread backward and outward
        v_angle_rad = torch.deg2rad(torch.tensor(self.cfg.swarm_cfg.formation_v_angle_deg, device=self.device))
        base_sep = self.cfg.swarm_cfg.formation_base_separation
        
        # Scale separation based on max_num_agents to ensure formation fits
        scale_factor = max(1.0, self.cfg.swarm_cfg.min_safe_distance / base_sep)
        effective_sep = base_sep * scale_factor
        
        # Generate formation positions template for each drone
        formation_template = torch.zeros(self.num_drones, 3, device=self.device)
        
        if self.num_drones == 1:
            # Single drone: centered at origin
            formation_template[0] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        else:
            # Multiple drones: inverted V formation
            # Apex drone (index 0) at the front (negative X)
            apex_idx = 0
            formation_template[apex_idx, 0] = self.cfg.swarm_cfg.formation_apex_offset  # X position (front)
            formation_template[apex_idx, 1] = 0.0  # Y position (centered)
            
            # Distribute remaining drones on left and right wings
            remaining_drones = self.num_drones - 1
            left_wing_count = remaining_drones // 2
            right_wing_count = remaining_drones - left_wing_count
            
            # Left wing (negative Y)
            for i in range(left_wing_count):
                wing_idx = i + 1
                x_offset = (i + 1) * effective_sep * torch.cos(v_angle_rad)  # Backward
                y_offset = -(i + 1) * effective_sep * torch.sin(v_angle_rad)  # Left
                formation_template[wing_idx, 0] = self.cfg.swarm_cfg.formation_apex_offset + x_offset
                formation_template[wing_idx, 1] = y_offset
            
            # Right wing (positive Y)
            for i in range(right_wing_count):
                wing_idx = left_wing_count + i + 1
                x_offset = (i + 1) * effective_sep * torch.cos(v_angle_rad)  # Backward
                y_offset = (i + 1) * effective_sep * torch.sin(v_angle_rad)  # Right
                formation_template[wing_idx, 0] = self.cfg.swarm_cfg.formation_apex_offset + x_offset
                formation_template[wing_idx, 1] = y_offset
    
        # Verify minimum separation constraint
        if self.num_drones > 1:
            dists = torch.cdist(formation_template.unsqueeze(0), formation_template.unsqueeze(0)).squeeze(0)
            # Set diagonal to large value to ignore self-distances
            dists = dists + torch.eye(self.num_drones, device=self.device) * 1000.0
            min_dist = dists.min()
            
            # If constraint violated, scale up the formation
            if min_dist < self.cfg.swarm_cfg.min_safe_distance:
                scale_up = self.cfg.swarm_cfg.min_safe_distance / min_dist
                formation_template[:, :2] *= scale_up
    
        # Expand template to all resetting environments
        # formation_template: (num_drones, 3)
        # Result: (num_reset_envs, num_drones, 3)
        formation_positions = formation_template.unsqueeze(0).expand(num_reset_envs, -1, -1).clone()
        
        # Add environment origins (XY) to all drones in each environment
        # env_origins: (num_reset_envs, 3)
        # Broadcast: (num_reset_envs, 1, 2) + (num_reset_envs, num_drones, 2)
        formation_positions[:, :, :2] += env_origins[:, :2].unsqueeze(1)
        
        # Add spawn heights (Z) to all drones in each environment
        # spawn_heights: (num_reset_envs,)
        # Broadcast: (num_reset_envs, 1) + (num_reset_envs, num_drones)
        formation_positions[:, :, 2] += spawn_heights.unsqueeze(1)
        
        return formation_positions  # Shape: (num_reset_envs, num_drones, 3)
    
