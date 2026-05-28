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
    ):
        self.env = env
        self.unwrapped_env = env.unwrapped

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

        self._make_specs()
        self._print_info(device, num_envs)

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
        self.observation_spec = Composite(
            agents=Composite(observation=obs_spec, shape=B),
            shape=B,
        )

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

        # --- centralized state spec -------------------------------------------
        if self.centralized_critic:
            self.state_spec = Composite(
                state=Unbounded(
                    shape=(*B, self.state_dim),
                    device=self.device, dtype=torch.float32,
                ),
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

    # -- env interface ---------------------------------------------------------
    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        obs_dict, _info = self.env.reset()

        # Stack per-agent observations → (num_envs, n_agents, obs_dim)
        obs_stacked = torch.stack(
            [obs_dict[a].to(self.device) for a in self.possible_agents], dim=1
        )

        td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        td["agents"] = TensorDict(
            {"observation": obs_stacked},
            batch_size=self.batch_size,
            device=self.device,
        )

        if self.centralized_critic:
            td["state"] = self.unwrapped_env._get_states().to(self.device)

        td["done"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        td["terminated"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        td["truncated"] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)

        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        # Extract per-agent actions from nested structure → flat dict for IsaacLab
        actions_stacked = tensordict["agents"]["action"]  # (num_envs, n_agents, action_dim)
        actions_dict = {}
        for i, agent_id in enumerate(self.possible_agents):
            actions_dict[agent_id] = actions_stacked[:, i, :]  # (num_envs, action_dim)

        obs_dict, rewards_dict, terminated_dict, truncated_dict, _info = self.env.step(actions_dict)

        # Stack observations → (num_envs, n_agents, obs_dim)
        obs_stacked = torch.stack(
            [obs_dict[a].to(self.device) for a in self.possible_agents], dim=1
        )
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
            td["state"] = self.unwrapped_env._get_states().to(self.device)

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

        return td

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
