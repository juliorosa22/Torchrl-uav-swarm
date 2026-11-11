"""
TorchRL wrapper for Isaac Lab DirectMARLEnv — shared-weight MAPPO.
Structures observations/rewards/actions in a nested "agents" TensorDict
so a single shared policy processes all agents in one forward pass.
Centralized critic state is kept at the top level.
"""

import torch
from typing import Optional
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite,
    Unbounded,
    Bounded,
    Categorical,
)
import gymnasium as gym

from isaaclab.envs import DirectMARLEnv
from UavSwarm.tasks.direct.torchrl_swarm.controller import (
    compute_baseline_action,
    compute_goal_bias,
    compute_swarm_gravity_baseline_action,
    compute_waypoint_plan_action,
)
try:
    # Newer rsl_rl: normalization.py lives under networks/.
    from rsl_rl.networks.normalization import EmpiricalNormalization
except ModuleNotFoundError:
    # Older rsl_rl (e.g. the lab GPU's pinned version): same module, under modules/ instead.
    from rsl_rl.modules.normalization import EmpiricalNormalization


class IsaacLabTorchRLWrapper(EnvBase):
    """Wraps Isaac Lab DirectMARLEnv for TorchRL shared-weight MAPPO.

    TensorDict structure (batch_size=[num_envs]):
        agents:
            observation   (num_envs, n_agents, obs_dim)
            action        (num_envs, n_agents, action_dim)
            sample_log_prob (num_envs, n_agents)
            reward        (num_envs, n_agents, 1)
        state              (num_envs, state_dim)   — centralized critic input
        done               (num_envs, 1)           — shared across agents
        terminated         (num_envs, 1)
        truncated          (num_envs, 1)

    The policy reads ("agents", "observation") and writes ("agents", "action").
    The critic reads ("state",) and writes ("state_value",).
    """

    def __init__(
        self,
        env: gym.Env,
        device: str = "cuda:0",
        centralized_critic: bool = True,
        normalize_obs: bool = False,
        residual_rl: bool = False,
        residual_baseline: str = "point",
        residual_kp: float = 2.0,
        residual_scale: float = 0.3,
        residual_repel_gain: float = 0.5,
        planner_range: float = 0.0,
        planner_alpha: float = 0.3,
        wp_horizon: int = 0,
        wp_dt: float = 0.2,
        wp_lookahead: float = 0.3,
        wp_safety: bool = True,
        wp_goal_bias: bool = False,
        wp_z_bias: bool = False,
    ):
        self.env = env
        self.unwrapped_env = env.unwrapped

        # Residual RL: env receives baseline(pos, desired_pos_w, ...) + residual_scale *
        # policy_action, not the raw policy action directly -- see controller.py's
        # compute_baseline_action ("point": same law proven in flight_test.py's
        # --goto_target mode) or compute_swarm_gravity_baseline_action ("apf": adds
        # inter-agent repulsion, for the swarm-gravity task). The policy's own sampled
        # action is left untouched in the tensordict (see _step) so ClipPPOLoss's
        # log-prob ratio stays correct -- PPO's distribution is over the residual, not
        # over the final env-facing action.
        if residual_baseline not in ("point", "apf"):
            raise ValueError(f"residual_baseline must be 'point' or 'apf', got {residual_baseline!r}")
        self.residual_rl = residual_rl
        self.residual_baseline = residual_baseline
        self.residual_kp = residual_kp
        self.residual_scale = residual_scale
        self.residual_repel_gain = residual_repel_gain
        self.planner_range = planner_range
        self.planner_alpha = planner_alpha
        if planner_range > 0.0 and not (residual_rl and residual_baseline == "apf"):
            raise ValueError("planner_range > 0 requires residual_rl with residual_baseline='apf'")
        self._planner_wp = None
        if wp_horizon > 0 and planner_range > 0.0:
            raise ValueError("wp_horizon and planner_range are mutually exclusive")
        self.wp_horizon = wp_horizon
        self.wp_dt = wp_dt
        self.wp_lookahead = wp_lookahead
        self.wp_safety = wp_safety
        self.wp_goal_bias = wp_goal_bias
        self.wp_z_bias = wp_z_bias
        self._wp_prev = None

        if not isinstance(self.unwrapped_env, DirectMARLEnv):
            raise TypeError(
                f"Environment must be DirectMARLEnv, got {type(self.unwrapped_env)}"
            )

        num_envs = self.unwrapped_env.num_envs
        super().__init__(device=device, batch_size=torch.Size([num_envs]))

        self.centralized_critic = centralized_critic
        self.possible_agents = self.unwrapped_env.possible_agents
        self.num_agents = len(self.possible_agents)

        # Extract per-agent space dimensions
        self.obs_spaces = {}
        self.action_spaces = {}
        for agent_id in self.possible_agents:
            self.obs_spaces[agent_id] = self.unwrapped_env.observation_spaces[agent_id]
            self.action_spaces[agent_id] = self.unwrapped_env.action_spaces[agent_id]

        self.obs_dim = self.obs_spaces[self.possible_agents[0]].shape[0]
        self.action_dim = self.action_spaces[self.possible_agents[0]].shape[0]

        # Centralized state dimension (all agent obs concatenated)
        dummy_state = self.unwrapped_env._get_states(dummy=True)
        self.state_dim = dummy_state.shape[1]

        # Running per-dimension obs normalization (opt-in -- default off so eval/play scripts
        # reusing this wrapper against checkpoints trained without it see unchanged behavior).
        # `state` is just per-agent observations concatenated (same 28-dim schema, num_agents
        # times), so it's normalized by tiling this same obs_rms rather than tracking separate
        # statistics for it -- one running estimate of "what does obs channel k look like",
        # not two that could drift apart.
        #
        # Backed by RSL-RL's EmpiricalNormalization (the reference obs normalizer used across
        # nearly all of Isaac Lab's own official RL tasks), not a hand-rolled Welford/SB3-
        # VecNormalize-style normalizer -- a from-scratch implementation (eps=1e-8, clip=+-10,
        # mathematically identical to VecNormalize's defaults) reliably produced NaN robot
        # velocities by the 2nd training iteration on this task (see
        # torchrl_mappo_cfg_remote.yaml's normalize_observations comment for the full story).
        # EmpiricalNormalization's smaller-magnitude eps and lack of clipping reliably avoided
        # the failure in side-by-side testing.
        self.normalize_obs = normalize_obs
        self.obs_rms = EmpiricalNormalization(shape=(self.obs_dim,), eps=1e-2).to(device) if normalize_obs else None

        self._env_obs_dim = self.obs_dim
        if self.wp_horizon > 0:
            self.obs_dim += 3 * self.wp_horizon
            self.state_dim += self.num_agents * 3 * self.wp_horizon
            self.action_dim = 3 * self.wp_horizon
            self._wp_prev = torch.zeros(num_envs, self.num_agents, self.action_dim, device=self.device)

        self._make_specs()
        self._print_info(device, num_envs)

        # Per-episode metrics the underlying env computes in _reset_idx (success rate,
        # formation error, collision rate, ...) and stores in extras["log"] -- SyncDataCollector
        # only sees the TensorDict from _step, so we capture them here and let the trainer
        # drain/average them once per rollout instead of losing them entirely.
        self.episode_logs: list[dict] = []

    # -- properties for the trainer to discover keys ---------------------------
    @property
    def reward_key(self):
        return ("agents", "reward")

    @property
    def action_key(self):
        return ("agents", "action")

    # -- spec construction -----------------------------------------------------
    def _make_specs(self):
        B = self.batch_size  # [num_envs]

        # --- observation spec -------------------------------------------------
        obs_spec = Unbounded(
            shape=(*B, self.num_agents, self.obs_dim),
            device=self.device,
            dtype=torch.float32,
        )
        obs_items = {"agents": Composite(observation=obs_spec, shape=B)}
        if self.centralized_critic:
            # Must be an observation_spec key: undeclared, the collector never advances "state" and the critic sees the reset-time state all episode.
            obs_items["state"] = Unbounded(shape=(*B, self.state_dim), device=self.device, dtype=torch.float32)
        self.observation_spec = Composite(**obs_items, shape=B)

        # --- action spec ------------------------------------------------------
        act_space = self.action_spaces[self.possible_agents[0]]
        low = float(act_space.low[0])
        high = float(act_space.high[0])
        self.action_spec = Composite(
            agents=Composite(
                action=Bounded(
                    low=low, high=high,
                    shape=(*B, self.num_agents, self.action_dim),
                    device=self.device, dtype=torch.float32,
                ),
                shape=B,
            ),
            shape=B,
        )

        # --- reward spec ------------------------------------------------------
        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(
                    shape=(*B, self.num_agents, 1),
                    device=self.device, dtype=torch.float32,
                ),
                shape=B,
            ),
            shape=B,
        )

        # --- done spec --------------------------------------------------------
        done_spec = Categorical(
            n=2, shape=(*B, 1), device=self.device, dtype=torch.bool,
        )
        self.done_spec = Composite(
            done=done_spec,
            terminated=done_spec.clone(),
            truncated=done_spec.clone(),
            shape=B,
        )

    def _print_info(self, device: str, num_envs: int):
        print(f"\n{'='*80}")
        print(f"[INFO] TorchRL Wrapper — {type(self.unwrapped_env).__name__}")
        print(f"{'='*80}")
        print(f"  Agents:           {self.num_agents} ({self.possible_agents})")
        print(f"  Parallel envs:    {num_envs}")
        print(f"  Obs dim (each):   {self.obs_dim}")
        print(f"  Action dim:       {self.action_dim}")
        print(f"  State dim (crit): {self.state_dim}")
        print(f"  Device:           {device}")
        print(f"{'='*80}\n")

    # -- observation normalization ----------------------------------------------
    def _normalize_obs_stacked(self, obs_stacked: torch.Tensor) -> torch.Tensor:
        """Update EmpiricalNormalization's running stats with this step's raw observations,
        then return the normalized copy. update() and forward() both expect (batch, obs_dim);
        obs_stacked is (num_envs, n_agents, obs_dim), so flatten the leading dims and restore
        the shape afterward.
        """
        assert self.obs_rms is not None
        n_envs, n_agents, obs_dim = obs_stacked.shape
        flat = obs_stacked.reshape(n_envs * n_agents, obs_dim)
        self.obs_rms.update(flat)
        normed = self.obs_rms(flat)
        return normed.reshape(n_envs, n_agents, obs_dim)

    def _get_state(self) -> torch.Tensor:
        """Centralized critic state -- per-agent observations concatenated, optionally
        followed by extra privileged dims a task appends (e.g. SwarmGravityV2's pairwise
        distance matrix -- see torchrl_swarm_env.py::_get_states). When obs normalization
        is on, only the leading num_agents*obs_dim portion is normalized, with the *same*
        obs_rms stats tiled across agents (that portion is literally num_agents copies of
        the same obs schema back to back, not a second independently-tracked statistic) --
        any trailing extra dims are left as-is, since they aren't obs-shaped.
        """
        state = self.unwrapped_env._get_states().to(self.device)
        obs_rms = self.obs_rms
        if self.normalize_obs and obs_rms is not None:
            obs_span = self.num_agents * self._env_obs_dim
            tiled_mean = obs_rms.mean.repeat(self.num_agents)
            tiled_std = obs_rms.std.repeat(self.num_agents)
            normed_obs_part = (state[..., :obs_span] - tiled_mean) / (tiled_std + obs_rms.eps)
            state = torch.cat([normed_obs_part, state[..., obs_span:]], dim=-1)
        return state

    def _get_state_aug(self) -> torch.Tensor:
        state = self._get_state()
        if self.wp_horizon > 0:
            state = torch.cat([state, self._wp_prev.reshape(state.shape[0], -1)], dim=-1)
        return state

    # -- env interface ---------------------------------------------------------
    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        obs_dict, _info = self.env.reset()

        # Stack per-agent observations → (num_envs, n_agents, obs_dim)
        obs_stacked = torch.stack(
            [obs_dict[a].to(self.device) for a in self.possible_agents], dim=1
        )
        if self.normalize_obs:
            obs_stacked = self._normalize_obs_stacked(obs_stacked)

        td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        td["agents"] = TensorDict(
            {"observation": obs_stacked},
            batch_size=self.batch_size,
            device=self.device,
        )

        if self.wp_horizon > 0:
            self._wp_prev.zero_()
            td["agents", "observation"] = torch.cat([obs_stacked, self._wp_prev], dim=-1)

        if self.centralized_critic:
            td["state"] = self._get_state_aug()

        td["done"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        td["terminated"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        td["truncated"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)

        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        # Extract per-agent actions from nested structure → flat dict for IsaacLab
        actions_stacked = tensordict["agents"]["action"]  # (num_envs, n_agents, action_dim)

        # Residual RL: the env receives baseline + residual_scale * policy_action, but
        # actions_stacked itself (used below only for the split into actions_dict) is
        # reassigned to that combined value -- the tensordict's own ("agents","action")
        # entry is untouched, so ClipPPOLoss's log-prob ratio still reads the policy's
        # actual sampled residual, not the env-facing combined action.
        env_actions = actions_stacked
        if self.wp_horizon > 0:
            K = self.wp_horizon
            raw_plan = actions_stacked.clamp(-1.0, 1.0)
            step_max = self.unwrapped_env.cfg.controller.max_lin_vel_cmd * self.wp_dt
            incr = raw_plan.reshape(*raw_plan.shape[:2], K, 3)
            if self.wp_goal_bias or self.wp_z_bias:
                bias = compute_goal_bias(self.unwrapped_env, self.residual_kp)
                if not self.wp_goal_bias:
                    bias = bias * torch.tensor([0.0, 0.0, 1.0], device=bias.device)
                incr = (incr + bias.unsqueeze(2)).clamp(-1.0, 1.0)
            plan_b = (step_max * incr).cumsum(dim=2)
            env_actions = compute_waypoint_plan_action(
                self.unwrapped_env, plan_b, self.wp_dt, self.wp_lookahead, self.wp_safety, self.residual_repel_gain
            )
            self._wp_prev = raw_plan.detach().clone()
        elif self.residual_rl and self.planner_range > 0.0:
            raw = actions_stacked[..., :3].clamp(-1.0, 1.0)
            if self._planner_wp is None:
                self._planner_wp = torch.zeros_like(raw)
            self._planner_wp = (1.0 - self.planner_alpha) * self._planner_wp + self.planner_alpha * raw
            env_actions = compute_swarm_gravity_baseline_action(
                self.unwrapped_env, self.residual_kp, self.residual_repel_gain,
                waypoint_offset_b=self.planner_range * self._planner_wp,
            )
        elif self.residual_rl:
            if self.residual_baseline == "apf":
                baseline = compute_swarm_gravity_baseline_action(
                    self.unwrapped_env, self.residual_kp, self.residual_repel_gain
                )
            else:
                baseline = compute_baseline_action(self.unwrapped_env, self.residual_kp)
            env_actions = (baseline + self.residual_scale * actions_stacked).clamp(-1.0, 1.0)

        actions_dict = {}
        for i, agent_id in enumerate(self.possible_agents):
            actions_dict[agent_id] = env_actions[:, i, :]  # (num_envs, action_dim)

        obs_dict, rewards_dict, terminated_dict, truncated_dict, _info = self.env.step(actions_dict)

        # Stack observations → (num_envs, n_agents, obs_dim)
        obs_stacked = torch.stack(
            [obs_dict[a].to(self.device) for a in self.possible_agents], dim=1
        )
        if self.normalize_obs:
            obs_stacked = self._normalize_obs_stacked(obs_stacked)
        # Stack rewards → (num_envs, n_agents, 1)
        reward_stacked = torch.stack(
            [
                rewards_dict[a].to(self.device).unsqueeze(-1)
                if rewards_dict[a].ndim == 1
                else rewards_dict[a].to(self.device)
                for a in self.possible_agents
            ],
            dim=1,
        )

        td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        td["agents"] = TensorDict(
            {"observation": obs_stacked, "reward": reward_stacked},
            batch_size=self.batch_size,
            device=self.device,
        )

        if self.centralized_critic:
            td["state"] = self._get_state_aug()

        # Aggregate done flags — episode done if ANY agent is done
        done = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        terminated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        truncated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)

        for agent_id in self.possible_agents:
            t = terminated_dict[agent_id].to(self.device)
            tr = truncated_dict[agent_id].to(self.device)
            if t.ndim == 1:
                t = t.unsqueeze(-1)
            if tr.ndim == 1:
                tr = tr.unsqueeze(-1)
            done = done | t | tr
            terminated = terminated | t
            truncated = truncated | tr

        td["done"] = done
        td["terminated"] = terminated
        td["truncated"] = truncated

        if self._planner_wp is not None:
            self._planner_wp[done.squeeze(-1)] = 0.0

        if self.wp_horizon > 0:
            self._wp_prev[done.squeeze(-1)] = 0.0
            td["agents", "observation"] = torch.cat([obs_stacked, self._wp_prev], dim=-1)
            if self.centralized_critic:
                td["state"] = self._get_state_aug()

        # extras["log"] is only freshly populated on a step where the env's own
        # _reset_idx ran (i.e. some env just terminated/truncated) -- gate on done so we
        # don't re-append the same stale dict on every subsequent step.
        if done.any():
            log = self.unwrapped_env.extras.get("log")
            if log:
                self.episode_logs.append(log)

        return td

    def drain_episode_logs(self) -> list[dict]:
        """Pop and clear all episode-end log dicts accumulated since the last drain."""
        logs, self.episode_logs = self.episode_logs, []
        return logs

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def make_torchrl_env(task: str, num_envs: int, device: str = "cuda:0", **env_kwargs):
    """Create Isaac Lab environment wrapped for TorchRL."""
    env = gym.make(task, num_envs=num_envs, **env_kwargs)
    return IsaacLabTorchRLWrapper(env, device=device)
