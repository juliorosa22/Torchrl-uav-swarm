"""Fully synthetic multi-agent point-mass navigation env -- no Isaac Sim.

Sanity-check env for scripts/torchrl/mappo_torchl.py's MAPPO trainer, decoupled from
Isaac Sim/physics entirely so a bad result can't be blamed on the simulator. Each agent
is a 2D point mass that must reach its own randomly-sampled goal. Deliberately mirrors the
real task's TensorDict contract (see IsaacLabTorchRLWrapper in torchrl_wrapper.py) exactly
-- individual per-agent reward, a centralized "state" that is literally every agent's
observation concatenated (so the same shared-V(s)-can't-explain-individual-rewards
ceiling applies here too) -- so a convergence result here transfers to a statement about
the MAPPO pipeline itself, not about a different problem.

Two kinematics modes, both behind the `momentum` flag (default off, preserving the
original behavior other variants/scripts were built against):
  momentum=False (default): action IS the velocity command -- pos += action*max_speed*dt.
    Agents can stop and change direction instantaneously; no way for control precision
    itself to be the reason a run fails to converge.
  momentum=True: action is an ACCELERATION command through linear drag -- a real
    double-integrator, closer to how a UAV's velocity actually responds to a commanded
    thrust vector than the instantaneous-teleport default. Tests whether momentum/drift
    making a tight simultaneous-arrival threshold hard to hit -- not the MARL/coordination
    structure -- is what's blocking the real Formation-TorchRL-UAVSwarm task. (The plain
    and V-formation variants of this env, in synthetic_formation_env.py, already both
    converged cleanly with momentum=False, ruling out the pipeline itself and the
    coordination structure as explanations.)
"""

import torch
from typing import Optional
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Composite, Unbounded, Bounded, Categorical


class SyntheticSwarmEnv(EnvBase):
    """TensorDict structure (batch_size=[num_envs]) -- identical shape to
    IsaacLabTorchRLWrapper:
        agents:
            observation   (num_envs, n_agents, obs_dim)   obs_dim = 4: rel_goal(2) + vel(2)
            action        (num_envs, n_agents, action_dim) action_dim = 2: velocity cmd in [-1,1]
            reward        (num_envs, n_agents, 1)
        state              (num_envs, n_agents * obs_dim)  -- concatenated per-agent obs
        done / terminated / truncated  (num_envs, 1)        -- shared across agents in an env
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int = 5,
        device: str = "cuda:0",
        max_episode_steps: int = 100,
        max_speed: float = 1.0,
        dt: float = 0.1,
        bound: float = 5.0,
        goal_threshold: float = 0.2,
        goal_bonus: float = 5.0,
        momentum: bool = False,
        max_accel: float = 3.0,
        drag: float = 0.5,
    ):
        super().__init__(device=device, batch_size=torch.Size([num_envs]))

        self.num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.max_episode_steps = max_episode_steps
        self.max_speed = max_speed
        self.dt = dt
        self.bound = bound
        self.goal_threshold = goal_threshold
        self.goal_bonus = goal_bonus
        self.momentum = momentum
        self.max_accel = max_accel
        self.drag = drag

        self.obs_dim = 4
        self.action_dim = 2
        self.state_dim = num_agents * self.obs_dim

        # Trainer/checkpoint code checks these two via getattr(env, ...) -- keep the
        # synthetic env's normalization surface identical (always off) rather than
        # omitting the attributes and relying on getattr's default every time.
        self.normalize_obs = False
        self.obs_rms = None

        n = num_envs
        a = num_agents
        self.pos = torch.zeros(n, a, 2, device=self.device)
        self.goal = torch.zeros(n, a, 2, device=self.device)
        self.vel = torch.zeros(n, a, 2, device=self.device)
        self.t = torch.zeros(n, dtype=torch.long, device=self.device)

        self._make_specs()
        self._print_info()

        self.episode_logs: list[dict] = []

    # -- properties for the trainer to discover keys ---------------------------
    @property
    def reward_key(self):
        return ("agents", "reward")

    @property
    def action_key(self):
        return ("agents", "action")

    # -- spec construction -------------------------------------------------
    def _make_specs(self):
        B = self.batch_size

        self.observation_spec = Composite(
            agents=Composite(
                observation=Unbounded(shape=(*B, self.num_agents, self.obs_dim), device=self.device, dtype=torch.float32),
                shape=B,
            ),
            shape=B,
        )
        self.action_spec = Composite(
            agents=Composite(
                action=Bounded(
                    low=-1.0, high=1.0,
                    shape=(*B, self.num_agents, self.action_dim),
                    device=self.device, dtype=torch.float32,
                ),
                shape=B,
            ),
            shape=B,
        )
        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(shape=(*B, self.num_agents, 1), device=self.device, dtype=torch.float32),
                shape=B,
            ),
            shape=B,
        )
        done_spec = Categorical(n=2, shape=(*B, 1), device=self.device, dtype=torch.bool)
        self.done_spec = Composite(
            done=done_spec, terminated=done_spec.clone(), truncated=done_spec.clone(), shape=B,
        )
        self.state_spec = Composite(
            state=Unbounded(shape=(*B, self.state_dim), device=self.device, dtype=torch.float32),
            shape=B,
        )

    def _print_info(self):
        print(f"\n{'='*80}")
        print(f"[INFO] SyntheticSwarmEnv (no Isaac Sim)")
        print(f"{'='*80}")
        print(f"  Agents:           {self.num_agents}")
        print(f"  Parallel envs:    {self.batch_size[0]}")
        print(f"  Obs dim (each):   {self.obs_dim}")
        print(f"  Action dim:       {self.action_dim}")
        print(f"  State dim (crit): {self.state_dim}")
        print(f"  Dynamics:         {'momentum (accel cmd + drag)' if self.momentum else 'direct (velocity cmd)'}")
        print(f"  Device:           {self.device}")
        print(f"{'='*80}\n")

    # -- internals -----------------------------------------------------------
    def _reset_kinematics(self, idx: torch.Tensor):
        """Zero velocity/step-counter for the given env indices. Shared by every goal-
        generation variant (SyntheticFormationEnv overrides _sample_spawn_and_goal but
        calls back into this for the kinematics-only part) so momentum support doesn't
        need to be reimplemented per subclass.
        """
        self.vel[idx] = 0.0
        self.t[idx] = 0

    def _sample_spawn_and_goal(self, mask: torch.Tensor):
        """Resample position/goal/velocity/step-counter for the envs where mask is True."""
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        n = idx.numel()
        self.pos[idx] = (torch.rand(n, self.num_agents, 2, device=self.device) * 2 - 1) * self.bound
        self.goal[idx] = (torch.rand(n, self.num_agents, 2, device=self.device) * 2 - 1) * self.bound
        self._reset_kinematics(idx)

    def _obs(self) -> torch.Tensor:
        return torch.cat([self.goal - self.pos, self.vel], dim=-1)  # (num_envs, n_agents, 4)

    def _state(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.reshape(obs.shape[0], self.state_dim)

    # -- env interface ---------------------------------------------------------
    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        all_envs = torch.ones(self.batch_size[0], dtype=torch.bool, device=self.device)
        self._sample_spawn_and_goal(all_envs)

        obs = self._obs()
        td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        td["agents"] = TensorDict({"observation": obs}, batch_size=self.batch_size, device=self.device)
        td["state"] = self._state(obs)
        td["done"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        td["terminated"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        td["truncated"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        actions = tensordict["agents", "action"].clamp(-1.0, 1.0)  # (num_envs, n_agents, 2)

        if self.momentum:
            # Double-integrator: action is an acceleration command, velocity carries over
            # between steps (linear drag, semi-implicit Euler), so the agent can't stop or
            # change direction instantaneously the way the default direct-control mode can.
            accel = actions * self.max_accel
            self.vel = self.vel * (1.0 - self.drag * self.dt) + accel * self.dt
            speed = self.vel.norm(dim=-1, keepdim=True)
            self.vel = self.vel * (self.max_speed / speed.clamp(min=self.max_speed))
            self.pos = self.pos + self.vel * self.dt
        else:
            # Direct control (original behavior): action IS the velocity command --
            # agents can stop/turn instantaneously, so control precision can't itself be
            # the reason convergence fails.
            self.vel = actions * self.max_speed
            self.pos = self.pos + self.vel * self.dt

        self.t += 1

        dist = torch.norm(self.goal - self.pos, dim=-1)  # (num_envs, n_agents)
        reached = dist < self.goal_threshold
        all_reached = reached.all(dim=-1)  # (num_envs,)
        timeout = self.t >= self.max_episode_steps  # (num_envs,)
        done_env = all_reached | timeout

        # Individual per-agent reward -- deliberately no cross-agent pooling, same
        # credit-assignment shape as the real task's get_formation_rewards_simple.
        reward = -dist + reached.float() * self.goal_bonus  # (num_envs, n_agents)

        obs = self._obs()

        if done_env.any():
            done_idx = done_env.nonzero(as_tuple=True)[0]
            for i in done_idx.tolist():
                self.episode_logs.append({
                    "Episode_Termination/goal_reached": float(all_reached[i].item()),
                    "Metrics/final_distance_to_goal": float(dist[i].mean().item()),
                })
            self._sample_spawn_and_goal(done_env)
            obs = self._obs()  # recompute so reset envs report their fresh obs this step

        state = self._state(obs)

        td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        td["agents"] = TensorDict(
            {"observation": obs, "reward": reward.unsqueeze(-1)},
            batch_size=self.batch_size, device=self.device,
        )
        td["state"] = state
        td["done"] = done_env.unsqueeze(-1)
        td["terminated"] = all_reached.unsqueeze(-1)
        td["truncated"] = (timeout & ~all_reached).unsqueeze(-1)
        return td

    def drain_episode_logs(self) -> list[dict]:
        logs, self.episode_logs = self.episode_logs, []
        return logs

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)

    def close(self):
        pass
