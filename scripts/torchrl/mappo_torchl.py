import torch
import time
import os
from typing import Optional
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn


class MAPPOPolicy(nn.Module):
    """Shared policy network — one instance processes all agents' observations."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(256, 256, 128)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor):
        """obs: (*batch, obs_dim) → mean (*batch, action_dim), std (*batch, action_dim)."""
        features = self.net(obs)
        mean = self.mean_layer(features)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std.expand_as(mean)


class CentralizedCritic(nn.Module):
    """Centralized critic — sees concatenated state of all agents."""

    def __init__(self, state_dim: int, hidden_sizes=(512, 512, 256)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor):
        """state: (*batch, state_dim) → value (*batch, 1)."""
        return self.net(state)


class MAPPO:
    """Shared-weight MAPPO trainer using TorchRL.

    Uses a single policy network for all agents by flattening the agent
    dimension into the batch dimension during PPO updates. The centralized
    critic sees the concatenated state of all agents.
    """

    def __init__(
        self,
        env,
        policy,
        critic,
        device: torch.device = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
        n_epochs: int = 10,
        batch_size: int = 64,
        n_agents: int = 5,
        frames_per_batch: int = 65536,
        model_name: str = "mappo_run",
        log_dir: Optional[str] = None,
        checkpoint_interval: int = 100,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.policy = policy
        self.critic = critic
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_interval = checkpoint_interval
        self.model_name = model_name

        # --- directories ---
        if log_dir is not None:
            self.run_dir = log_dir
        else:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            self.run_dir = os.path.join("logs", "torchrl", f"{model_name}_{time_str}")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- collector ---
        self.collector = SyncDataCollector(
            env,
            policy,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            device=self.device,
            reset_at_each_iter=True,
        )

        # --- replay buffer ---
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            batch_size=batch_size,
        )

        # --- Flat-key critic for flattened TD (reads "state" at root) ---
        self.critic_flat = TensorDictModule(
            module=self.critic.module,
            in_keys=["state"],
            out_keys=["state_value"],
        )

        # --- PPO loss (uses ClipPPOLoss DEFAULT keys: "reward", "action", "done", etc.) ---
        # After _flatten_agents, all keys are at top level matching ClipPPOLoss defaults.
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.critic_flat,
            clip_epsilon=clip_epsilon,
            entropy_bonus=True,
            loss_critic_type="l2",
            entropy_coeff=c2,
            critic_coeff=c1,
            normalize_advantage=False,
            safe=True,
        )
        self.loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=gae_lambda)
        self.gae = self.loss_module.value_estimator
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr=lr)

        # --- logging ---
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "logs"))
        self.last_avg_reward = 0.0

        print(f"[INFO] MAPPO trainer ready — checkpoints: {self.checkpoint_dir}")

    # ------------------------------------------------------------------
    def train(self, total_frames: int):
        pbar = tqdm(total=total_frames, unit="frames", desc="Training")
        collected_frames = 0

        for i, rollout in enumerate(self.collector):
            if collected_frames >= total_frames:
                break

            # Remove collector metadata
            rollout.pop("collector", None)

            # --- Step 1: Flatten (frames, envs, agents) → single batch ---
            rollout_flat = _flatten_agents(rollout, self.n_agents)

            # --- Step 2: GAE on flattened data (computes state_value internally) ---
            with torch.no_grad():
                self.gae(rollout_flat)

            # Store flat rollout in buffer
            self.buffer.extend(rollout_flat)

            # --- PPO update epochs ---
            total_obj, total_critic, total_ent = 0.0, 0.0, 0.0
            total_kl, total_clip_frac, total_ess, total_ev, total_entropy = 0.0, 0.0, 0.0, 0.0, 0.0
            total_grad_norm = 0.0
            num_updates = 0

            for _ in range(self.n_epochs):
                for _ in range(self.collector.frames_per_batch // self.batch_size):
                    mini_batch = self.buffer.sample().to(self.device)
                    loss_vals = self.loss_module(mini_batch)
                    loss = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    total_obj += loss_vals["loss_objective"].item()
                    total_critic += loss_vals["loss_critic"].item()
                    total_ent += loss_vals["loss_entropy"].item()
                    total_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                    # PPO diagnostics (ClipPPOLoss computes these; extract for logging)
                    for key, accum in [
                        ("kl_approx", "total_kl"),
                        ("clip_fraction", "total_clip_frac"),
                        ("ESS", "total_ess"),
                        ("explained_variance", "total_ev"),
                        ("entropy", "total_entropy"),
                    ]:
                        if key in loss_vals.keys():
                            locals()[accum] += loss_vals[key].item()

                    num_updates += 1

            # Sync collector policy
            self.collector.update_policy_weights_()

            # --- checkpoint ---
            if (i + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(i + 1)

            # --- logging ---
            frames = rollout.numel()
            collected_frames += frames
            pbar.update(frames)

            avg_obj = total_obj / max(num_updates, 1)
            avg_critic = total_critic / max(num_updates, 1)
            avg_ent = total_ent / max(num_updates, 1)
            avg_kl = total_kl / max(num_updates, 1)
            avg_clip_frac = total_clip_frac / max(num_updates, 1)
            avg_ess = total_ess / max(num_updates, 1)
            avg_ev = total_ev / max(num_updates, 1)
            avg_entropy = total_entropy / max(num_updates, 1)
            avg_grad_norm = total_grad_norm / max(num_updates, 1)

            reward_tensor = rollout_flat["next", "reward"]
            self.last_avg_reward = reward_tensor.mean().item()
            reward_std = reward_tensor.std().item()

            self.writer.add_scalar("Loss/Policy", avg_obj, collected_frames)
            self.writer.add_scalar("Loss/Value", avg_critic, collected_frames)
            self.writer.add_scalar("Loss/Entropy", avg_ent, collected_frames)
            self.writer.add_scalar("Loss/Total", avg_obj + avg_critic + avg_ent, collected_frames)
            self.writer.add_scalar("Reward/Average", self.last_avg_reward, collected_frames)
            self.writer.add_scalar("Reward/Std", reward_std, collected_frames)

            self.writer.add_scalar("Diagnostics/kl_approx", avg_kl, collected_frames)
            self.writer.add_scalar("Diagnostics/clip_fraction", avg_clip_frac, collected_frames)
            self.writer.add_scalar("Diagnostics/ESS", avg_ess, collected_frames)
            self.writer.add_scalar("Diagnostics/explained_variance", avg_ev, collected_frames)
            self.writer.add_scalar("Diagnostics/entropy", avg_entropy, collected_frames)
            self.writer.add_scalar("Diagnostics/grad_norm", avg_grad_norm, collected_frames)

            pbar.set_postfix({
                "reward": f"{self.last_avg_reward:.2f}",
                "loss": f"{avg_obj + avg_critic + avg_ent:.4f}",
                "kl": f"{avg_kl:.4f}",
                "clip": f"{avg_clip_frac:.2f}",
                "ev": f"{avg_ev:.3f}",
            })

        pbar.close()
        self.save_checkpoint(collected_frames)
        print(f"[INFO] Training complete — {collected_frames:,} frames")

    def save_checkpoint(self, iteration: int):
        torch.save(
            {"policy": self.policy.state_dict(), "critic": self.critic.state_dict()},
            os.path.join(self.checkpoint_dir, f"checkpoint_{self.model_name}_iter_{iteration}.pt"),
        )
        print(f"[Checkpoint] Saved at iteration {iteration}")


def _flatten_agents(td: TensorDict, n_agents: int) -> TensorDict:
    """Flatten (frames, envs, agents) → single batch for shared-weight PPO.

    Input TensorDict (from collector) has batch_size=[T] where T = frames_per_batch.
    Each value's leading dims are [T, num_envs, ...].
    Agents subtree entries have shape [T, num_envs, n_agents, ...].

    Output has batch_size=[T * num_envs * n_agents] with agent keys at root level
    AND re-nested under "agents" for policy in_keys compatibility. Centralized values
    (state_value, advantage, etc.) are repeated n_agents times so every
    (frame, env, agent) sample has its own copy.
    """
    T = td.batch_size[0]
    num_envs = td["agents", "observation"].shape[1]
    total = T * num_envs * n_agents
    obs_dim = td["agents", "observation"].shape[-1]

    # -- agent-level keys: reshape from [T, num_envs, n_agents, ...] to [total, ...] --
    def _flatten_agent_val(val):
        return val.reshape(total, *val.shape[3:])

    agents = td.pop("agents")
    result = TensorDict({}, batch_size=torch.Size([total]), device=td.device)

    for key in agents.keys(include_nested=True, leaves_only=True):
        result[key] = _flatten_agent_val(agents[key])

    # -- centralized keys: repeat per agent, then flatten --
    def _expand_centralized(val):
        v = val.unsqueeze(2).expand(T, num_envs, n_agents, *val.shape[2:])
        return v.reshape(total, *val.shape[2:])

    for key in ("done", "terminated", "truncated", "state_value", "advantage", "value_target", "state"):
        if key in td.keys():
            result[key] = _expand_centralized(td.pop(key))

    # -- re-nest agent keys under "agents" for policy in_keys compatibility --
    # TensorDict stores references, so nested and flat keys share the same tensors.
    _agent_leaf_keys = list(agents.keys(include_nested=True, leaves_only=True))
    result["agents"] = TensorDict(
        {k: result[k] for k in _agent_leaf_keys if k in result.keys()},
        batch_size=torch.Size([total]),
        device=td.device,
    )

    # -- next subtree --
    if "next" in td.keys():
        nxt = td.pop("next")
        next_result = TensorDict({}, batch_size=torch.Size([total]), device=td.device)

        _next_agent_keys = []
        if "agents" in nxt.keys():
            nxt_agents = nxt.pop("agents")
            _next_agent_keys = list(nxt_agents.keys(include_nested=True, leaves_only=True))
            for key in _next_agent_keys:
                next_result[key] = _flatten_agent_val(nxt_agents[key])

        for key in ("done", "terminated", "truncated", "state_value", "state"):
            if key in nxt.keys():
                next_result[key] = _expand_centralized(nxt.pop(key))

        # Reconstruct centralized state from next-agent observations for GAE
        if "observation" in next_result.keys() and "state" not in next_result.keys():
            next_result["state"] = (
                next_result["observation"]
                .reshape(T, num_envs, n_agents, obs_dim)
                .reshape(T, num_envs, n_agents * obs_dim)
                .unsqueeze(2).expand(T, num_envs, n_agents, n_agents * obs_dim)
                .reshape(total, n_agents * obs_dim)
            )

        # Re-nest next agent keys for policy in_keys compatibility
        if _next_agent_keys:
            next_result["agents"] = TensorDict(
                {k: next_result[k] for k in _next_agent_keys if k in next_result.keys()},
                batch_size=torch.Size([total]),
                device=td.device,
            )

        result["next"] = next_result

    return result
