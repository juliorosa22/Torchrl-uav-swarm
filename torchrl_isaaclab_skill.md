# TorchRL + IsaacLab Integration — Claude Code Skill Reference

> **Purpose**: This document teaches Claude Code how to correctly use TorchRL components with IsaacLab (NVIDIA's GPU-accelerated robotics simulation platform). Use it as a reference when writing, reviewing, or debugging RL training code that combines TorchRL and IsaacLab.

---

## Overview

IsaacLab environments are **pre-vectorized** — a single `gym.make` call spins up thousands of parallel environments on the GPU (e.g., 4096). This changes how standard TorchRL patterns apply. The key integration point is `IsaacLabWrapper`.

---

## 1. Wrapping the Environment

Use `IsaacLabWrapper` to convert a gymnasium IsaacLab environment into a TorchRL-compatible `EnvBase`:

```python
import gymnasium as gym
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

env = gym.make("Isaac-Ant-v0", cfg=env_cfg)
env = IsaacLabWrapper(env)
```

### Key Defaults

| Parameter | Default Value | Why It Matters |
|-----------|--------------|----------------|
| `device` | `cuda:0` | IsaacLab runs natively on GPU |
| `allow_done_after_reset` | `True` | IsaacLab can report `done=True` right after reset |
| `convert_actions_to_numpy` | `False` | Actions remain as tensors, avoiding CPU round-trips |

### Important Behavioral Notes

- **In-place tensor mutation**: IsaacLab modifies `terminated` and `truncated` tensors in-place. `IsaacLabWrapper` automatically clones these to prevent data corruption.
- **Batched specs**: IsaacLab environment specs include the batch dimension (e.g., shape `(4096, obs_dim)`). Use `*_spec_unbatched` properties when you need per-environment shapes.
- **Reward shape**: IsaacLab rewards have shape `(num_envs,)`. The wrapper unsqueezes to `(num_envs, 1)` for TorchRL compatibility.

---

## 2. Data Collection

Because IsaacLab environments are pre-vectorized, use a **single `Collector`** — no `ParallelEnv` or `MultiCollector` needed:

```python
from torchrl.collectors import Collector

collector = Collector(
    create_env_fn=env,
    policy=policy,
    frames_per_batch=40960,  # 10 env steps × 4096 envs
    storing_device="cpu",
    no_cuda_sync=True,       # CRITICAL for CUDA envs — prevents mysterious hangs
)
```

- **`no_cuda_sync=True`**: Avoids unnecessary CUDA synchronization that causes hangs with GPU-native environments. **Always set this.**
- **`storing_device="cpu"`**: Moves collected data to CPU for the replay buffer, freeing GPU memory.

### 2-GPU Async Pipeline (Maximum Throughput)

Split simulation and training across two GPUs using a background collection thread:

- **GPU 0 (`sim_device`)**: IsaacLab simulation + collection policy inference
- **GPU 1 (`train_device`)**: Model training (world model, actor, value gradients)

```python
import copy
import threading
from tensordict import TensorDict

# Deep-copy the policy to sim_device for collection
collector_policy = copy.deepcopy(policy).to(sim_device)

# Background thread: continuous data collection
def collect_loop(collector, replay_buffer, stop_event):
    for data in collector:
        replay_buffer.extend(data)
        if stop_event.is_set():
            break

# Main thread: training on train_device
for optim_step in range(total_steps):
    batch = replay_buffer.sample()
    train(batch)  # all on cuda:1

    # Periodic weight sync: training policy → collector policy
    if optim_step % sync_every == 0:
        weights = TensorDict.from_module(policy)
        collector.update_policy_weights_(weights)
```

**Key points for 2-GPU setup:**
- Both CUDA operations release the GIL, so they truly overlap.
- Always pass `TensorDict.from_module(policy)` — not the module itself — to `update_policy_weights_()`.
- Set `CUDA_VISIBLE_DEVICES=0,1` to expose both GPUs (IsaacLab defaults to GPU 0 only).
- Falls back gracefully to single-GPU if only one GPU is available.

### Distributed Collection with RayCollector

For multi-GPU or multi-node setups:

```python
from torchrl.collectors.distributed import RayCollector

collector = RayCollector(
    [make_env] * num_collectors,
    policy,
    frames_per_batch=8192,
    collector_kwargs={
        "trust_policy": True,
        "no_cuda_sync": True,
    },
)
```

---

## 3. Replay Buffer

Use `SliceSampler` for sequential trajectory sampling. It requires enough contiguous data:

```
init_random_frames >= batch_length × num_envs
                    = 50 × 4096
                    = 204,800
```

For GPU-resident replay buffers, use `LazyTensorStorage` with the target CUDA device to avoid CPU→GPU transfers at sample time (transfers happen at `extend` time instead).

---

## 4. Known Gotchas & Fixes

These are critical issues that commonly appear when combining TorchRL and IsaacLab:

### Gotcha 1 — `no_cuda_sync=True` (Always Required)

Always set `no_cuda_sync=True` for collectors with CUDA environments. Without it, you get mysterious hangs with no clear error message.

### Gotcha 2 — Installing TorchRL in Isaac Container

```bash
pip install torchrl --no-build-isolation --no-deps
```

Use `--no-build-isolation --no-deps` to avoid conflicts with Isaac's pre-installed torch/numpy.

### Gotcha 3 — `TensorDictPrimer` and `expand_specs`

When adding primers (e.g., `state`, `belief`) to a pre-vectorized environment, you **must** pass `expand_specs=True` to `TensorDictPrimer`. Otherwise, the primer shape `()` conflicts with the environment's `batch_size` of `(4096,)`.

```python
from torchrl.envs.transforms import TensorDictPrimer

primer = TensorDictPrimer(
    primers={"state": state_spec},
    expand_specs=True,  # REQUIRED for pre-vectorized envs
)
```

### Gotcha 4 — Model-Based Env Spec Double-Batching

`model_based_env.set_specs_from_env(batched_env)` copies specs with batch dimensions baked in. The model-based env then double-batches actions (e.g., `(4096, 4096, 8)` instead of `(4096, 8)`).

**Fix**: Unbatch the model-based env's specs after copying:

```python
model_based_env.set_specs_from_env(test_env)

if test_env.batch_size:
    idx = (0,) * len(test_env.batch_size)
    model_based_env.__dict__["_output_spec"] = (
        model_based_env.__dict__["_output_spec"][idx]
    )
    model_based_env.__dict__["_input_spec"] = (
        model_based_env.__dict__["_input_spec"][idx]
    )
    model_based_env.empty_cache()
```

### Gotcha 5 — `torch.compile` with TensorDict

Compiling full loss modules crashes because `dynamo` traces through TensorDict internals.

**Fix**: Compile only individual MLP sub-modules (encoder, decoder, reward_model, value_model):

```python
import torch._dynamo

torch._dynamo.config.suppress_errors = True

encoder = torch.compile(encoder)
decoder = torch.compile(decoder)
reward_model = torch.compile(reward_model)
value_model = torch.compile(value_model)

# Do NOT compile RSSM (sequential, shared with collector)
# Do NOT compile loss modules (heavy TensorDict use)
```

### Gotcha 6 — `SliceSampler` with `strict_length=False`

The sampler may return fewer elements than `batch_size`, causing `reshape(-1, batch_length)` to fail.

**Fix**: Truncate the sample before reshaping:

```python
sample = replay_buffer.sample()
numel = sample.numel()
usable = (numel // batch_length) * batch_length

if usable < numel:
    sample = sample[:usable]

sample = sample.reshape(-1, batch_length)
```

### Gotcha 7 — `frames_per_batch` vs `batch_length`

Each collection adds `frames_per_batch / num_envs` time steps per environment. The `SliceSampler` needs contiguous sequences of at least `batch_length` steps within a single trajectory.

**Rule of thumb**:
```
frames_per_batch >= batch_length × num_envs
```
Or ensure `init_random_frames >= batch_length × num_envs`.

### Gotcha 8 — `TD_GET_DEFAULTS_TO_NONE` Environment Variable

When running inside the Isaac container, set this variable to ensure correct TensorDict default behavior:

```bash
export TD_GET_DEFAULTS_TO_NONE=1
```

Or in Python:
```python
import os
os.environ["TD_GET_DEFAULTS_TO_NONE"] = "1"
```

---

## 5. Quick-Reference Checklist

When writing TorchRL + IsaacLab code, verify:

- [ ] `IsaacLabWrapper` is used to wrap the gymnasium env
- [ ] Single `Collector` is used (not `ParallelEnv`)
- [ ] `no_cuda_sync=True` is set on the collector
- [ ] `init_random_frames >= batch_length × num_envs`
- [ ] `TensorDictPrimer` uses `expand_specs=True`
- [ ] `torch.compile` is only applied to MLP sub-modules, not loss modules or RSSM
- [ ] `SliceSampler` samples are truncated before reshaping if `strict_length=False`
- [ ] `TD_GET_DEFAULTS_TO_NONE=1` is set in the Isaac container
- [ ] For 2-GPU setups: weights sync uses `TensorDict.from_module(policy)` and `CUDA_VISIBLE_DEVICES=0,1`

---

*Source: [TorchRL IsaacLab Integration Docs](https://docs.pytorch.org/rl/main/reference/isaaclab.html)*
