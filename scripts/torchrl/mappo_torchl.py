import torch
import time
import os
import math
from typing import Optional
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn


class RunningMeanStd:
    """Scalar running mean/variance via Welford/Chan's parallel-variance combination
    (same algorithm as OpenAI Baselines' RunningMeanStd) -- numerically stable, no need to
    store samples.
    """

    def __init__(self, device=None, epsilon: float = 1e-4):
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = epsilon

    def update(self, x: torch.Tensor):
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = x.numel()

        # A single non-finite sample here would permanently poison self.mean/self.var --
        # every later update's formula folds in the previous mean/var, so NaN propagates
        # forever once it enters, silently breaking reward normalization (and therefore
        # every downstream loss) for the rest of the run without ever raising. Skip
        # rather than absorb; the env's own reward already clamps NaN/Inf to 0 (see
        # get_formation_rewards's nan_to_num), so this is a last-resort guard, not the
        # primary defense.
        if not (torch.isfinite(batch_mean) and torch.isfinite(batch_var)):
            return

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count


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
        normalize_advantage: bool = True,
        normalize_rewards: bool = True,
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
        self.gamma = gamma

        # --- reward normalization (standard PPO stabilizer; see train()'s _normalize_rewards_
        # -inplace for why this -- not a raw value-target rescale -- is what's implemented) ---
        self.normalize_rewards = normalize_rewards
        if self.normalize_rewards:
            num_envs = env.batch_size[0]
            self.returns = torch.zeros(num_envs, n_agents, device=self.device)
            self.reward_rms = RunningMeanStd(device=self.device)

        # --- directories ---
        if log_dir is not None:
            self.run_dir = log_dir
        else:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            self.run_dir = os.path.join("logs", "torchrl", f"{model_name}_{time_str}")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- collector ---
        # reset_at_each_iter=False: episodes run to their natural termination across
        # multiple collection calls. True would hard-cap every episode at
        # frames_per_batch/num_envs steps, preventing the policy from ever seeing
        # states beyond the first few seconds of an episode.
        self.collector = SyncDataCollector(
            env,
            policy,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            device=self.device,
            reset_at_each_iter=False,
        )

        # After _flatten_agents the buffer holds T * num_envs * n_agents transitions,
        # not just frames_per_batch. Sizing at frames_per_batch alone would silently
        # discard (n_agents - 1)/n_agents ≈ 80% of each rollout.
        self._buffer_size = frames_per_batch * n_agents

        # --- replay buffer ---
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self._buffer_size),
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
            # torchrl_mappo_cfg.yaml declares normalize_advantages: true but that value was
            # never actually read (mappo_train.py never passes it through) -- this was
            # hardcoded False regardless, contradicting the config's stated intent. Now wired
            # through from config; batch-level advantage normalization is standard PPO practice.
            normalize_advantage=normalize_advantage,
            # PPO2-style value clipping: bounds how far the critic's prediction can move
            # from its pre-update value in a single minibatch step (loss = max of the
            # unclipped and clipped squared error), same clip_epsilon as the policy's
            # trust region. Added after reward normalization alone still let Loss/Value
            # explode 3.3 -> 54M over 100 iterations (grad_norm up to 27M) -- the critic
            # loss was so much larger than the policy loss that the shared gradient clip
            # left almost nothing for the policy (clip_fraction/kl_approx both collapsed
            # toward 0 by iteration 100, i.e. the policy had effectively stopped updating).
            clip_value=True,
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

            # --- Step 0: reward normalization (stabilizes the critic target scale; see
            # _normalize_rewards_inplace) -- must run before GAE so bootstrapping and the
            # critic loss both operate on the same normalized reward scale end to end.
            if self.normalize_rewards:
                self._normalize_rewards_inplace(rollout)

            # --- Step 1: Flatten (frames, envs, agents) → single batch ---
            rollout_flat = _flatten_agents(rollout, self.n_agents)

            # --- Step 2: GAE on flattened data (computes state_value internally) ---
            with torch.no_grad():
                self.gae(rollout_flat)

            # Store flat rollout in buffer
            self.buffer.extend(rollout_flat)

            # --- PPO update epochs ---
            total_obj, total_critic, total_ent = 0.0, 0.0, 0.0
            total_grad_norm = 0.0
            num_updates = 0
            num_skipped_updates = 0
            # NOTE: previously accumulated via `locals()[accum] += ...` in the loop below --
            # locals() returns a snapshot dict in CPython; writing into it does not affect
            # the actual local variables, so those diagnostics were always frozen at 0.0
            # regardless of what ClipPPOLoss returned (confirmed: kl/clip/ESS/explained_variance
            # stayed exactly 0.0000 for an entire 100-iteration validation run). A plain dict
            # accumulator (mutated in place, not reassigned) has no such issue.
            diag_totals = {"kl_approx": 0.0, "clip_fraction": 0.0, "ESS": 0.0, "explained_variance": 0.0, "entropy": 0.0}

            for _ in range(self.n_epochs):
                for _ in range(self._buffer_size // self.batch_size):
                    mini_batch = self.buffer.sample().to(self.device)
                    loss_vals = self.loss_module(mini_batch)
                    loss = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Rare PPO numerical instability (e.g. an extreme action/value estimate
                    # pushing the loss to NaN/Inf) must not reach optimizer.step() -- once
                    # network weights go NaN they never recover on their own (every later
                    # forward/backward pass stays NaN), silently wasting the rest of the run.
                    # Confirmed necessary: seed 1 of the first 5-seed sweep hit exactly this
                    # at 11% through and produced NaN losses for the remainder undetected --
                    # train_multi_seed.py doesn't treat it as a failure since the process
                    # itself never crashes. Skip the minibatch entirely instead.
                    if not torch.isfinite(loss):
                        num_skipped_updates += 1
                        self.optimizer.zero_grad()
                        continue

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), 1.0)

                    if not torch.isfinite(grad_norm):
                        num_skipped_updates += 1
                        self.optimizer.zero_grad()
                        continue

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    total_obj += loss_vals["loss_objective"].item()
                    total_critic += loss_vals["loss_critic"].item()
                    total_ent += loss_vals["loss_entropy"].item()
                    total_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                    # PPO diagnostics (ClipPPOLoss computes these; extract for logging)
                    for key in diag_totals:
                        if key in loss_vals.keys():
                            diag_totals[key] += loss_vals[key].item()

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
            avg_kl = diag_totals["kl_approx"] / max(num_updates, 1)
            avg_clip_frac = diag_totals["clip_fraction"] / max(num_updates, 1)
            avg_ess = diag_totals["ESS"] / max(num_updates, 1)
            avg_ev = diag_totals["explained_variance"] / max(num_updates, 1)
            avg_entropy = diag_totals["entropy"] / max(num_updates, 1)
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
            self.writer.add_scalar("Diagnostics/skipped_updates", num_skipped_updates, collected_frames)
            if num_skipped_updates > 0:
                print(f"[WARN] Skipped {num_skipped_updates} minibatch update(s) this iteration "
                      f"(non-finite loss/grad_norm) -- see Diagnostics/skipped_updates")
            if self.normalize_rewards:
                self.writer.add_scalar("Diagnostics/reward_running_std", self._last_reward_std, collected_frames)

            # Task-level episode metrics (success rate, formation error, collision rate, ...)
            # computed by the env's own _reset_idx and drained from the wrapper. Averaged
            # across every episode that ended during this rollout window.
            episode_logs = self.env.drain_episode_logs()
            success_rate = None
            if episode_logs:
                keys = set().union(*(d.keys() for d in episode_logs))
                for key in keys:
                    vals = [d[key] for d in episode_logs if key in d]
                    self.writer.add_scalar(key, sum(vals) / len(vals), collected_frames)
                    if key == "Episode_Termination/goal_reached":
                        success_rate = sum(vals) / len(vals)

            # Action channel statistics — track per-dim mean/std to confirm the policy
            # is exploring the full velocity command range (dims 0-2: vx/vy/vz, dim 3: yaw_rate).
            if "action" in rollout_flat.keys():
                actions = rollout_flat["action"]  # (total, action_dim)
                labels = ["action_ch0", "action_ch1", "action_ch2", "action_ch3"]
                for idx, label in enumerate(labels):
                    if idx < actions.shape[-1]:
                        self.writer.add_scalar(f"Actions/{label}_mean", actions[..., idx].mean().item(), collected_frames)
                        self.writer.add_scalar(f"Actions/{label}_std", actions[..., idx].std().item(), collected_frames)

            postfix = {
                "reward": f"{self.last_avg_reward:.2f}" if math.isfinite(self.last_avg_reward) else "nan!",
                "loss": f"{avg_obj + avg_critic + avg_ent:.4f}" if math.isfinite(avg_obj) else "nan!",
                "kl": f"{avg_kl:.4f}",
                "clip": f"{avg_clip_frac:.2f}",
                "ev": f"{avg_ev:.3f}",
            }
            if success_rate is not None:
                postfix["succ"] = f"{success_rate:.2f}"
            pbar.set_postfix(postfix)

        pbar.close()
        self.save_checkpoint(collected_frames)
        print(f"[INFO] Training complete — {collected_frames:,} frames")

    def _normalize_rewards_inplace(self, rollout: TensorDict) -> None:
        """Divide rewards by a running estimate of the per-env discounted-return std
        (OpenAI Baselines / CleanRL-style reward normalization), in place, before GAE runs.

        This -- not normalizing the value target directly (PopArt) -- is what's implemented
        for the yaml's `normalize_values` intent: it needs no separate raw/normalized critic
        output split, since GAE bootstraps (`reward + gamma * V(s')`) on the same normalized
        reward scale throughout rather than mixing raw rewards with rescaled values. Confirmed
        necessary empirically: a validation run without this showed Loss/Value growing
        ~1000x (2.4k -> 2.27M) over 100 iterations with the pre-clip grad norm exploding to
        ~218k, consistent with the critic regressing onto unbounded raw returns.
        """
        # Layout is (num_envs, T, n_agents, 1) -- env-dim first, time second (verified
        # empirically via SyncDataCollector; _flatten_agents's own "T = batch_size[0]"
        # naming is misleading here, since its correctness only depends on the total
        # element count, not on which leading dim is actually time vs. env).
        rewards = rollout["next", "agents", "reward"]  # (num_envs, T, n_agents, 1)
        dones = rollout["next", "done"].float()  # (num_envs, T, 1)
        T = rewards.shape[1]
        normed = torch.empty_like(rewards)
        std = torch.sqrt(self.reward_rms.var + 1e-8)
        for t in range(T):
            r_t = rewards[:, t, :, 0]  # (num_envs, n_agents)
            d_t = dones[:, t]  # (num_envs, 1), broadcasts over n_agents
            self.returns = self.returns * self.gamma * (1.0 - d_t) + r_t
            self.reward_rms.update(self.returns.reshape(-1))
            std = torch.sqrt(self.reward_rms.var + 1e-8)
            normed[:, t, :, 0] = r_t / std
        rollout["next", "agents", "reward"] = normed
        self._last_reward_std = float(std.item())

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
