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
        max_grad_norm: float = 1.0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.policy = policy
        self.critic = critic
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_interval = checkpoint_interval
        # torchrl_mappo_cfg.yaml declares max_grad_norm but it was never threaded through --
        # clip_grad_norm_ below used a bare hardcoded 1.0 regardless of the config's value.
        self.max_grad_norm = max_grad_norm
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

        # After _finalize_flat the buffer holds T * num_envs * n_agents transitions,
        # not just frames_per_batch. Sizing at frames_per_batch alone would silently
        # discard (n_agents - 1)/n_agents ≈ 80% of each rollout.
        self._buffer_size = frames_per_batch * n_agents

        # --- replay buffer ---
        # NOTE: tried SamplerWithoutReplacement here (shuffled, non-overlapping per-epoch
        # partition -- canonical PPO practice, vs. the default RandomSampler's with-
        # replacement draws) to close a known fidelity gap. Reverted: reproduced the same
        # NaN-velocity instability EmpiricalNormalization was just confirmed to fix, 2/2
        # local runs, despite having no direct mechanism to touch the environment/physics --
        # it only changes which stored transitions get selected for gradient updates. Most
        # likely explanation: it alters the training trajectory enough (different transitions
        # weighted into early gradient steps) to produce a differently-behaved policy that
        # still triggers whatever underlying fragility causes physics divergence -- possibly
        # the SE(3) controller's frame-construction singularity (see compute_geometric_
        # controller's b2_des cross-product, which can degenerate when desired thrust
        # direction and desired heading go near-collinear). Revisit once that's confirmed/
        # ruled out; don't re-add without a smoke test run showing it's actually clean.
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
        # After _finalize_flat, all keys are at top level matching ClipPPOLoss defaults.
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.critic_flat,
            clip_epsilon=clip_epsilon,
            entropy_bonus=True,
            # Huber loss (TorchRL's own default -- this was previously overridden to "l2"),
            # not squared error: L2 loss grows QUADRATICALLY with the critic's residual, so
            # once it overshoots by any real margin, the loss and gradient for that sample
            # scale up disproportionately, pushing the next update further off -- a
            # self-reinforcing spiral. Directly observed on a 5M-frame run: Loss/Value sat
            # in a healthy 5-40 range for ~1M frames (iterations 84-118), then, with no
            # external trigger (reward_running_std stayed smooth through the same window),
            # entered a runaway that roughly doubled every few iterations for the rest of
            # training (64 -> 5,921 -> 25,078 -> 67,394) -- the same shape as the 3.3 -> 54M
            # explosion clip_value was added for below, just recurring despite it. Huber
            # matches L2 near the optimum (same useful gradient signal for typical-sized
            # errors) but grows only LINEARLY beyond a threshold, breaking the feedback loop
            # instead of just bounding how far one update can go (clip_value, still useful
            # as a second line of defense, kept below).
            loss_critic_type="smooth_l1",
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

            # --- Step 1: Reshape into per-(env,agent) trajectories, keeping T explicit ---
            # so GAE's recursive bootstrap walks real time, not an arbitrary flattened
            # order (see _reshape_for_gae's docstring for why this used to be broken).
            reshaped = _reshape_for_gae(rollout, self.n_agents)

            # --- Step 2: GAE — genuinely time-aware now (computes state_value internally) ---
            with torch.no_grad():
                self.gae(reshaped)

            # --- Diagnostic: can a single per-env V(s) even explain individual per-agent
            # targets? Law of total variance: Var(target) = E[Var(target|env,t)] +
            # Var(E[target|env,t]). The first term is variance ACROSS AGENTS at fixed
            # (env,t) -- CentralizedCritic's "state" input (and thus its prediction) is
            # identical for every agent within one env/timestep (see _reshape_for_gae's
            # _centralized_to_gae_shape), so a shared V(s) can never reduce this term no
            # matter how well trained. It caps explained_variance at (1 - this fraction)
            # regardless of critic quality. Tests whether the persistently-near-zero
            # Diagnostics/explained_variance (unmoved by every fix so far) is this
            # structural ceiling from the individual-reward fix, not a tuning problem.
            with torch.no_grad():
                num_envs_g = reshaped.batch_size[0] // self.n_agents
                T_g = reshaped.batch_size[1]

                def _agent_variance_fraction(flat: torch.Tensor) -> float:
                    x = flat.reshape(num_envs_g, self.n_agents, T_g)
                    total_var = x.var(unbiased=False).item()
                    if total_var < 1e-12:
                        return 0.0
                    within_agent_var = x.var(dim=1, unbiased=False).mean().item()
                    return within_agent_var / total_var

                reward_agent_frac = _agent_variance_fraction(reshaped["next", "reward"])
                value_target_agent_frac = _agent_variance_fraction(reshaped["value_target"])

            # --- Step 3: collapse to a flat batch for the PPO minibatch loop / buffer ---
            rollout_flat = _finalize_flat(reshaped)

            # Store flat rollout in buffer
            self.buffer.extend(rollout_flat)

            # --- PPO update epochs ---
            total_obj, total_critic, total_ent = 0.0, 0.0, 0.0
            total_actor_grad_norm = 0.0
            total_critic_grad_norm = 0.0
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
                    # Separate clips per network -- a single combined clip over both networks'
                    # parameters lets whichever one has the larger raw gradient (almost always
                    # the critic here: deeper net, value_loss_coef=1.0, L2 loss on unbounded
                    # returns) dictate the shared scaling factor, silently shrinking the
                    # actor's effective step far below what its own gradient would warrant on
                    # its own. Confirmed on the formation-simple-reward diagnostic run: KL and
                    # clip_fraction stayed ~0.002-0.004 / ~1-3% for the entire 1M-frame run
                    # (policy essentially frozen, entropy flat) while Loss/Value kept dropping
                    # and outweighed Loss/Policy by 4-5 orders of magnitude -- exactly the
                    # collapse already flagged above (clip_value's docstring) as a known risk
                    # of clipping both networks together.
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                    if not (torch.isfinite(actor_grad_norm) and torch.isfinite(critic_grad_norm)):
                        num_skipped_updates += 1
                        self.optimizer.zero_grad()
                        continue

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    total_obj += loss_vals["loss_objective"].item()
                    total_critic += loss_vals["loss_critic"].item()
                    total_ent += loss_vals["loss_entropy"].item()
                    total_actor_grad_norm += actor_grad_norm.item()
                    total_critic_grad_norm += critic_grad_norm.item()

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
            avg_actor_grad_norm = total_actor_grad_norm / max(num_updates, 1)
            avg_critic_grad_norm = total_critic_grad_norm / max(num_updates, 1)

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
            self.writer.add_scalar("Diagnostics/reward_agent_variance_fraction", reward_agent_frac, collected_frames)
            self.writer.add_scalar("Diagnostics/value_target_agent_variance_fraction", value_target_agent_frac, collected_frames)
            self.writer.add_scalar("Diagnostics/explained_variance_ceiling", 1.0 - value_target_agent_frac, collected_frames)
            self.writer.add_scalar("Diagnostics/entropy", avg_entropy, collected_frames)
            self.writer.add_scalar("Diagnostics/actor_grad_norm", avg_actor_grad_norm, collected_frames)
            self.writer.add_scalar("Diagnostics/critic_grad_norm", avg_critic_grad_norm, collected_frames)
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
        # empirically via SyncDataCollector; matches _reshape_for_gae's own layout
        # assumption below).
        rewards = rollout["next", "agents", "reward"]  # (num_envs, T, n_agents, 1)
        dones = rollout["next", "done"].float()  # (num_envs, T, 1)
        T = rewards.shape[1]
        normed = torch.empty_like(rewards)
        std = torch.sqrt(self.reward_rms.var + 1e-8)
        for t in range(T):
            r_t = rewards[:, t, :, 0]  # (num_envs, n_agents)
            d_t = dones[:, t]  # (num_envs, 1), broadcasts over n_agents
            self.returns = self.returns * self.gamma * (1.0 - d_t) + r_t
            # NaN*0 is still NaN (IEEE754), so a single bad reward would otherwise poison
            # this env's running return forever, even across the done-reset above -- and
            # from there the shared reward_rms permanently, since it never resets itself.
            self.returns = torch.nan_to_num(self.returns, nan=0.0, posinf=0.0, neginf=0.0)
            self.reward_rms.update(self.returns.reshape(-1))
            std = torch.sqrt(self.reward_rms.var + 1e-8)
            normed[:, t, :, 0] = r_t / std
        rollout["next", "agents", "reward"] = normed
        self._last_reward_std = float(std.item())

    def save_checkpoint(self, iteration: int):
        ckpt = {"policy": self.policy.state_dict(), "critic": self.critic.state_dict()}
        # Obs normalization stats live on the env wrapper, not this trainer -- a resumed run
        # that reloads policy/critic weights but starts obs_rms fresh (mean=0/var=1) would feed
        # the already-converged policy a differently-scaled input than it was trained on.
        # EmpiricalNormalization is an nn.Module (mean/var/std/count are registered buffers),
        # so its own state_dict() is the correct save/load mechanism.
        if getattr(self.env, "normalize_obs", False):
            ckpt["obs_rms"] = self.env.obs_rms.state_dict()
        torch.save(
            ckpt,
            os.path.join(self.checkpoint_dir, f"checkpoint_{self.model_name}_iter_{iteration}.pt"),
        )
        print(f"[Checkpoint] Saved at iteration {iteration}")


def _reshape_for_gae(td: TensorDict, n_agents: int) -> TensorDict:
    """Reshape a collector rollout (num_envs, T, ...) into (num_envs*n_agents, T, ...) --
    merging env+agent into one batch dimension while keeping T as its own, explicit axis.

    GAE's recursive lambda-return bootstrap needs a genuine time dimension to walk (its
    own docstring: input must be shaped [*B, T], defaulting to "the last dimension" as
    time when none is named). The previous single-shot flatten collapsed
    (num_envs, T, n_agents) into ONE flat dimension *before* calling GAE, silently
    destroying that structure -- GAE's "last dimension is time" fallback then bootstrapped
    across whatever order the flattening happened to interleave samples in (different
    agents, then different envs), not real consecutive timesteps of one trajectory.
    Confirmed with a synthetic test: constant per-(env,agent) value + zero reward should
    give ~0 advantage everywhere; the old flatten produced advantages up to +-110 by
    bootstrapping across completely unrelated envs' value scales. This was upstream of
    every PPO update in every experiment this session -- reward shape, gradient clipping,
    observation normalization, replay-buffer sampling -- none of which touch this step.

    Collector layout (verified empirically, not assumed): agent-level keys are shaped
    (num_envs, T, n_agents, *F); centralized keys (state, done, terminated, truncated)
    are (num_envs, T, *F) with no agent dimension -- termination/state are env-wide, not
    per-agent, in this task (confirmed in torchrl_wrapper.py: done is the OR of every
    agent's terminated/truncated, shared across the whole env).

    Returns a TensorDict with batch_size=[num_envs*n_agents, T] -- current-step keys at
    root (flat, not yet re-nested under "agents"), and a "next" subtree of the same shape.
    Call self.gae(...) directly on this output; only flatten further (see
    _finalize_flat) once GAE has added "advantage"/"value_target".
    """
    num_envs = td.batch_size[0]
    T = td["agents", "observation"].shape[1]
    B = num_envs * n_agents

    def _agent_to_gae_shape(val: torch.Tensor) -> torch.Tensor:
        # (num_envs, T, n_agents, *F) -> (num_envs, n_agents, T, *F) -> (B, T, *F)
        v = val.permute(0, 2, 1, *range(3, val.ndim))
        return v.reshape(B, T, *val.shape[3:])

    def _centralized_to_gae_shape(val: torch.Tensor) -> torch.Tensor:
        # (num_envs, T, *F) -> (num_envs, n_agents, T, *F) -> (B, T, *F): same value
        # repeated per agent (state/done/terminated are env-wide, not agent-specific).
        v = val.unsqueeze(1).expand(num_envs, n_agents, T, *val.shape[2:])
        return v.reshape(B, T, *val.shape[2:])

    agents = td.pop("agents")
    result = TensorDict({}, batch_size=torch.Size([B, T]), device=td.device)
    for key in agents.keys(include_nested=True, leaves_only=True):
        result[key] = _agent_to_gae_shape(agents[key])

    for key in ("done", "terminated", "truncated", "state"):
        if key in td.keys():
            result[key] = _centralized_to_gae_shape(td.pop(key))

    if "next" in td.keys():
        nxt = td.pop("next")
        next_result = TensorDict({}, batch_size=torch.Size([B, T]), device=td.device)

        if "agents" in nxt.keys():
            nxt_agents = nxt.pop("agents")
            for key in nxt_agents.keys(include_nested=True, leaves_only=True):
                next_result[key] = _agent_to_gae_shape(nxt_agents[key])

        for key in ("done", "terminated", "truncated", "state"):
            if key in nxt.keys():
                next_result[key] = _centralized_to_gae_shape(nxt.pop(key))

        # SyncDataCollector doesn't thread state_spec-declared keys into "next" the same
        # way it does observation_spec keys -- confirmed empirically: torchrl_wrapper.py's
        # _step() sets td["state"] on every call, but rollout["next"] arrives without
        # "state" regardless. Reconstruct it the same way the env itself does (concatenate
        # all agents' next observations, per env) rather than silently running GAE with a
        # missing next-state.
        if "state" not in next_result.keys() and "observation" in next_result.keys():
            obs_dim = next_result["observation"].shape[-1]
            per_env_agent = next_result["observation"].reshape(num_envs, n_agents, T, obs_dim)
            concatenated = per_env_agent.permute(0, 2, 1, 3).reshape(num_envs, T, n_agents * obs_dim)
            next_result["state"] = _centralized_to_gae_shape(concatenated)

        result["next"] = next_result

    return result


def _finalize_flat(td: TensorDict) -> TensorDict:
    """Collapse a (B=num_envs*n_agents, T)-shaped, GAE-processed rollout into a single
    flat batch dimension for the PPO minibatch loop / replay buffer, and re-nest
    per-agent keys under "agents" for the actor's in_keys.

    Safe to fully flatten now: GAE has already consumed the temporal structure it
    needed (advantage/value_target are already computed), and PPO's clipped objective
    treats every (env, agent, t) sample independently regardless of order.
    """
    B, T = td.batch_size
    total = B * T

    agent_keys = [k for k in ("observation", "action", "sample_log_prob") if k in td.keys()]
    next_agent_keys = (
        [k for k in ("observation", "reward") if k in td["next"].keys()]
        if "next" in td.keys() else []
    )

    flat = td.reshape(total)

    flat["agents"] = TensorDict(
        {k: flat[k] for k in agent_keys},
        batch_size=torch.Size([total]),
        device=td.device,
    )
    if next_agent_keys:
        flat["next", "agents"] = TensorDict(
            {k: flat["next", k] for k in next_agent_keys},
            batch_size=torch.Size([total]),
            device=td.device,
        )

    return flat
