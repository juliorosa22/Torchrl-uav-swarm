# Plan: imitation start, critic pre-training, MAPPO fine-tuning (moving-sphere packing)

Status as of 2026-11-15. Written as shared context: it explains the concepts the pipeline relies on, what has been measured, what is built, and what is still planned. Items marked **planned** are not implemented or not yet run.

## 1. Task and metrics

- Swarm of N Crazyflies spawns packed inside a containment sphere whose centre travels a path. Drones must stay inside the sphere and keep nearest-neighbour distance >= 0.5 m while it moves, then settle for a 6 s hold.
- Sphere radius R = (min_safe_distance/2)(N/0.6)^(1/3): about 1.01 m at N=5, 1.61 m at N=20.
- A step is **packed** if all drones are inside the sphere and the nearest-neighbour distance is >= 0.5 m. Episode **success** = no collision/out-of-bounds and packed fraction >= 0.9. The headline number is the mean **packed fraction**; success is also reported by speed bin (1-1.5 / 1.5-2 / 2-2.5 / 2.5-3 m/s).
- Paths: families line, curve, slalom, arc, hairpin, spline (train); figure-eight, roller coaster, helix (held out). Speed 1-3 m/s, lengths 20 or 30 m, profiles const / ramp / stop-and-go.
- Sensing tier used for RL: **lidar** (82 beams, 3 m, noise, 10 Hz, one-step delay; hits carry no identity or velocity).
- Evaluation protocol: 288 episodes on all 9 families (`apf_tune.py`, seed 0). Every number below uses it.

## 2. Concepts

- **APF (artificial potential field).** Hand-built controller: attraction to the sphere centre (position pull plus feed-forward of the sphere velocity), repulsion from nearby lidar returns, velocity damping. The **preview** variant looks ahead on the path (attraction to the centre `tp` s ahead; feed-forward = mean sphere velocity over `tf` s). Gains are tuned per N by evolution strategy on packed fraction.
- **Action.** Per drone 4 numbers in [-1,1]: body-frame velocity command (vx, vy, vz) and yaw rate, scaled by the 4 m/s cap, tracked by a geometric controller. The APF always commands yaw rate 0. `--yaw_hold` removes the yaw output (policy has 3 outputs, yaw rate fixed at 0).
- **Observation per drone** (obs_dim = 30 + 5k): own body-frame velocity, angular velocity, projected gravity (9); offset to sphere centre / R (3), sphere velocity (3), R (1), hold flag (1); path preview offsets at 0.5/1.0/1.5 s (9); previous action (4); k lidar tokens (range/3 m, body-frame direction, valid flag). k = 8 for N=5, 12 for N>=10. The tokens are the k nearest beam returns.
- **Imitation learning (IL).** Supervised regression of a policy onto an expert's actions. Plain behaviour cloning trains on expert-visited states only, so small errors push the policy into states it has never seen (compounding error).
- **DAgger.** Iterative fix for that: run the current policy (mixed with the expert with probability beta, decaying), record the states it actually visits, label them with the expert's action, aggregate everything, retrain. Here the expert is the preview APF reading the same lidar scan as the policy.
- **MAPPO.** PPO with a shared actor and a **centralised critic** V(s) that sees the concatenated observations of all drones plus privileged positions/velocities relative to the sphere. Actions are TanhNormal: a Gaussian on a pre-tanh mean with learned log-std, squashed to [-1,1]. Evaluation uses tanh(mean).
- **GAE and reward normalisation.** Advantages use GAE (gamma 0.99, lambda 0.95). Rewards are divided by a running std of discounted returns (`reward_rms`); the critic therefore regresses onto returns in that normalised scale. `reward_rms` is not stored in checkpoints.
- **Shared critic across drones.** One V(s) per environment is used for all drones, so it cannot explain per-drone reward differences within an environment; this caps explained variance below 1.

## 3. What has been measured

Packed fraction, 288 episodes, lidar, mixed paths.

| System | N=5 | N=20 | Notes |
|---|---|---|---|
| Plain tuned APF | 0.911 | 0.807 | |
| Preview APF (tuned per N) | 0.938 | 0.804 | N=10: 0.913; no collisions; spacing weak at N=20 (min NN 0.43 m) |
| DAgger imitation, pure policy | 0.938 (round 11) | 0.807 (round 15) | success by bin N=5: 100/99/89/32 %; N=20: 41/32/15/11 %; no collisions |
| Pure RL, velocity output (9.8M frames) | 0.263 | - | collides in 100% of episodes, median episode 1.7 s |
| Gain-mod RL (5 multipliers) | best 0.899, then collapsed | 0.70 | started from mismatched gains |
| Scale RL arm A (2 outputs, current reward), 4.9M frames | 0.927 | - | below its own starting point 0.938; arm B (hinge reward) pending |
| Residual RL on APF (3 runs), waypoint RL | no gain | no gain | earlier sessions |

Findings that drive this plan:

1. **Observation, action space and network are sufficient.** Imitation matches the expert at both N with k=8 and k=12 tokens and a 3-output velocity action. RL's failures are not a representation limit (single seed, tested at N=5 and N=20).
2. **The reward is the metric.** On the preview APF at N=20, reward per step is 0.333; across episodes it correlates 0.999 with packed fraction. Terms per step: packed bonus +0.405, centring -0.068, smoothness -0.0016, containment -0.002, **spacing -0.0004**. The spacing penalty is about 1000x smaller than the shared binary packed bonus, so a drone gets almost no signal about spacing violations.
3. **Early-training signal is weak.** Pure RL has no gradient to avoid collisions in the first seconds; RL arms that start at the APF could not find improvements.

## 4. Pipeline

```
 expert APF (tuned preview gains, lidar)
        |
        v
 [Stage 1] DAgger -> actor checkpoint (3-output attention policy)         BUILT, run at N=5 and N=20
 [Stage 2] critic value pre-training on the same rollouts                  PLANNED
        |   -> checkpoint {"policy", "critic", reward_rms}
        v
 [Stage 3] MAPPO fine-tuning at N=20 (low noise, frozen-actor critic warm-up, hinge reward)   PARTLY BUILT
        |
        v
 [Stage 4] evaluation vs preview APF, 288 episodes, >= 3 seeds             BUILT (single seed so far)
```

### Stage 1: DAgger (built)

`scripts/torchrl/apf_tune.py --dagger ROUNDS --zero_policy --final_params KP REPEL KD FF RSCALE --preview_params TP TF --obs_sense lidar --obs_k K`. Each round: `--il_steps` env steps with expert mixing probability beta = 0.5^round for rounds < 4, then 0; every visited (observation, expert action) pair is stored (ring buffer `--il_cap`); then `--il_fit` minibatch updates (4096, Adam 1e-3) of MSE between tanh(policy mean) and the clamped expert action (3 dims). Writes `<il_out>_r<round>.pt` in the trainer's checkpoint format (`{"policy": ...}`, policy log-std stays 0).

Runs: N=5, 12 rounds x 1000 steps, 32 envs; N=20, 16 rounds x 300 steps, 32 envs, k=12, cap 3.2M samples. Both finish in about 20 minutes on the local 6 GB GPU. Evaluate a checkpoint with `--pure_policy --yaw_hold --policy_ckpt <file>` (no APF in the loop).

### Stage 2: critic pre-training (planned)

Goal: give MAPPO a critic that already predicts the imitation policy's returns, so the first PPO updates are not driven by noise.

- Same `CentralizedCritic` architecture and centralised state as the trainer (the loader already reads `ckpt["critic"]`).
- Data: rollouts of the imitation policy with zero expert mixing, run with the **fine-tune reward** (`--space_hinge`) and the **fine-tune exploration noise** (sampled actions at log-std -2) so targets describe the policy PPO will start from.
- Targets: per-environment discounted Monte-Carlo return (gamma 0.99, running return reset at episode ends, exactly as the trainer accumulates it), averaged over drones because V is shared. Monte-Carlo targets avoid bootstrapping from an untrained critic, which is why this converges faster than the online warm-up.
- Scale: divide targets by the running std of discounted returns computed from the same rollouts; save it as `reward_rms` and load it into the trainer so critic output and normalised rewards match.
- Hold out some environments and report explained variance; do not unfreeze the actor below the gate in section 6.
- Must be done at the target N (critic input size depends on N and k).
- A short online warm-up (about 10 iterations) still follows, to absorb pipeline differences.

### Stage 3: MAPPO fine-tuning (partly built)

Flags in `mappo_train.py` (built): `--checkpoint`, `--reset_log_std`, `--critic_warmup_iters` (actor frozen), `--policy_lr_scale`, `--lr`, `--space_hinge DSEP POW`, `--yaw_hold`, `--entropy_coef`. Missing-critic checkpoints load with a fresh critic.

Starting settings (proposed): N=20, start from the N=20 imitation checkpoint (+ pre-trained critic once Stage 2 exists), `--reset_log_std -2.0`, `--entropy_coef 0`, `--policy_lr_scale 0.05` (policy lr 1.5e-5), reward = hinge penalty starting at 0.6 m, linear, 600 iterations, 32 envs, 8192 frames per batch, checkpoint every 12 iterations, evaluate at several checkpoints.

Why these values: a smoke test (N=5) showed KL 0.0 while the actor was frozen and KL 2.2 on the first iteration after unfreezing with a barely trained critic (explained variance -29). At log-std -3 (std 0.05) a mean shift of 0.05 already gives KL of about 0.5, so the policy learning rate must be far lower than the default and the critic must be ready first.

### Stage 4: evaluation

Packed fraction, success by speed bin, median min NN, fraction of steps below 0.5 m, collisions, mean reward per step (apf_tune rows now carry `reward` and per-term `rew_terms`). Final claims need >= 3 training seeds and held-out families reported separately.

## 5. Reward variants

- **A (current).** Spacing penalty = sum over neighbours of ((0.55 - d)/0.55)^2 for d < 0.55; shared +0.5 bonus when packed; -10 on collision/out-of-bounds; centring -0.1 d/R; containment -(d-R)+/R; smoothness -0.05 |da|^2.
- **B (`--space_hinge 0.6 1`).** Spacing penalty linear from 0.6 m: about 0.17 per neighbour per step at 0.5 m, 0.33 at 0.4 m. Result of the scale arm with B is pending.
- Env code: `moving_task.py` (`RW`, `get_rewards`; `env._rew_terms` holds per-term means).

## 6. Decision gates (proposed)

1. Imitation within 0.01 of the expert's packed fraction at the target N. **Met** at N=5 (0.938 vs 0.938) and N=20 (0.807 vs 0.804).
2. Critic: explained variance on held-out environments high enough that the first PPO iterations do not move the policy erratically (target about 0.8 or better; reconsider if the shared-critic ceiling is lower). Do not unfreeze before this.
3. Fine-tune: KL stays near the 0.02 target, no collapse, and N=20 packed fraction beats the preview APF (0.804) by a margin that exceeds seed noise (>= 3 seeds). If it does not, RL is reported as a negative result and the paper is the benchmark plus tuned baselines.
4. If fine-tuning helps, test zero-shot transfer to N=30/40 against the preview-APF scaling curve (requires tokens with k fixed at 12 and a critic re-initialised per N).

## 7. Risks and open questions

- Policy and critic can be unstable at the moment the actor is unfrozen; mitigation is the long critic phase and the low learning rate.
- Beating the APF may be capped by what lidar sensing allows; at N=5 the APF is already at 0.938 and the room is small. N=20 (settle phase, fast bins) is the realistic target.
- The expert is tuned per N; imitation inherits that tuning, so zero-shot scaling claims need care.
- Imitation was evaluated with one seed; the multi-seed evaluation is still to do.
- Checkpoints from earlier gain-mod runs were evaluated with a clamped mean rather than tanh(mean); this was fixed in commit 9ffec7d, and earlier gain-mod numbers were not re-scored.

## 8. Code map

| Item | Location |
|---|---|
| DAgger, pure-policy evaluation, reward logging | `scripts/torchrl/apf_tune.py` (`--dagger`, `--il_*`, `--pure_policy`, `--yaw_hold`) |
| Training entry, checkpoint resume, fine-tune flags | `scripts/torchrl/mappo_train.py` |
| Trainer (actor and critic param groups, critic warm-up) | `scripts/torchrl/mappo_torchl.py` |
| Wrapper (yaw hold, gain modulation) | `scripts/torchrl/torchrl_wrapper.py` |
| Observation, reward | `source/UavSwarm/UavSwarm/tasks/direct/torchrl_swarm/moving_task.py` |
| APF controllers (plain, preview, gain-mod) | `source/UavSwarm/UavSwarm/tasks/direct/torchrl_swarm/controller.py` |
| Lidar model | `source/UavSwarm/UavSwarm/tasks/direct/torchrl_swarm/lidar.py` |
| Related work | `docs/related_work.md` |

Tuned preview gains (kp, repel, kd, ff, rscale | tp, tf): N=5 1.345, 3.652, 2.562, 0.511, 0.949 | 0.343, 0.676; N=20 0.7917, 1.9914, 4.4067, 1.0837, 1.2644 | 0.1622, 0.8586.

Machines: local RTX 3060 6 GB (small runs, DAgger, smoke tests); lab desktop RTX A4000 16 GB via `lab-gpu` (long runs; one Isaac Sim per GPU; Tailscale login approval needed). Lab jobs need a clean pushed tree (`scripts/remote_run.sh`).
