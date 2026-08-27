# Future Work: BenchMARL Integration

Not started. Scoped 2026-08-27 as a follow-up track, not a blocker for the current
formation-scalability training run. Revisit once the formation task is validated (converges
at N=5, scalability sweep shows sensible trends) — integrating a benchmarking framework
around a task that doesn't work yet just makes debugging harder.

## Why

Two payoffs beyond "easier to evaluate":

1. **More algorithms for a broader comparison.** BenchMARL ships 9 algorithms out of the box
   (MAPPO, IPPO, MADDPG, QMIX, VDN, ...) — a broader comparison than our hand-rolled MAPPO
   trainer (`scripts/torchrl/mappo_torchl.py`) supports today.
2. **Built-in statistical reporting.** BenchMARL integrates `marl-eval` (IQM, bootstrapped
   confidence intervals, performance profiles) — the same methodology
   `scripts/torchrl/aggregate_results.py` reimplements by hand (self-contained, no `rliable`
   dependency, built 2026-08-27 for the multi-seed formation-scalability runs). If we
   integrate BenchMARL later, that script mostly gets superseded rather than wasted — the
   metric definitions and the "why IQM/CI" reasoning carry over directly.

## Why it's more tractable than it looks

`IsaacLabTorchRLWrapper` (`scripts/torchrl/torchrl_wrapper.py`) is **already a TorchRL
`EnvBase`** with fully-built specs (`_make_specs()`) — exactly BenchMARL's integration
contract. BenchMARL requires a `Task` enum (one entry per task, config auto-loaded from a
matching yaml) plus a `TaskClass` implementing ~9 methods
(see [BenchMARL's task-extension example](https://github.com/facebookresearch/BenchMARL/tree/main/examples/extending/task)):

| BenchMARL `TaskClass` method | What it needs | What we already have |
|---|---|---|
| `get_env_fun(num_envs, ...)` | `Callable[[], EnvBase]` | Wrap the existing `gym.make(task_id, cfg=...)` -> `IsaacLabTorchRLWrapper(...)` construction from `mappo_train.py` into a lambda. `num_envs` maps onto `cfg.scene.num_envs` -- Isaac Lab is natively GPU-vectorized into one batched `EnvBase`, the same vectorization model BenchMARL expects for "vectorized native environments". |
| `observation_spec` / `action_spec` / `state_spec` | `CompositeSpec` per group | Already built in `torchrl_wrapper.py::_make_specs()` -- near-direct passthrough (`env.observation_spec`, `env.action_spec`, `env.state_spec`). |
| `group_map` | `{"agents": [...]}` | Matches our wrapper's existing key structure (`possible_agents` list). |
| `supports_continuous_actions` | bool | `True` -- action space is `Bounded[-1,1]`, no discrete case exists. |
| `max_steps` | int | `env.max_episode_length` -- already a property on the Isaac Lab env. |
| `has_render`, `action_mask_spec`, `info_spec` | usually `False`/`None` | Not applicable, trivial stubs. |
| `log_info(batch)` | optional extra logging | Could surface the `extras["log"]` dict already wired through (success rate, formation error, `swarm_cohesion`) into BenchMARL's own logger. |

Net new code: roughly one file shaped like BenchMARL's `common.py` template
(~100-150 lines) plus one `conf/task/uavswarm/formation.yaml`. Not a rewrite of the
environment -- it's already in the right shape.

## The one real friction point

Isaac Sim's `AppLauncher` must boot **before** any `isaaclab` import, exactly once per
process. BenchMARL's normal entry point (`python benchmarl/run.py task=... algorithm=...`)
is a Hydra CLI that doesn't know about this -- Hydra would try to instantiate the task
(which imports `isaaclab`) before we get a chance to launch the sim app.

**Fix**: skip the Hydra CLI, drive BenchMARL programmatically from a small custom launcher
script (same shape as `mappo_train.py` today -- boot `AppLauncher` first, *then* build a
BenchMARL `Experiment` object in code and call `.run()`). This is the standard pattern other
Isaac-Lab-style TorchRL integrations use for the same reason.

## Rough integration checklist (when this gets picked up)

1. New file, e.g. `source/UavSwarm/UavSwarm/tasks/direct/torchrl_swarm/benchmarl_task.py`:
   `UavSwarmTask(Task)` enum (`FORMATION = None`, ...) + `UavSwarmTaskClass(TaskClass)`
   implementing the table above.
2. `conf/task/uavswarm/formation.yaml` -- task config (num_agents, controller type, stage,
   episode length, etc.), matching `FormationUAVSwarmEnvCfg` defaults.
3. New launcher script, e.g. `scripts/torchrl/benchmarl_train.py` -- boots `AppLauncher`,
   registers `UavSwarmTask`, builds and runs a BenchMARL `Experiment` in code (no Hydra CLI).
4. Verify multi-seed + `marl-eval` output end-to-end on the formation task before treating it
   as the primary benchmarking path; keep `aggregate_results.py` as a fallback until then.
5. Only after that: consider swapping in additional algorithms (IPPO, MADDPG, QMIX, VDN) for
   a broader comparison beyond MAPPO.
