  How the parameters interact in TorchRL

  Looking at mappo_torchl.py:

  1. SyncDataCollector collects exactly frames_per_batch frames per iteration
  (line 102-109). With 256 envs, that's frames_per_batch / 256 steps per env.
  2. Training loop (line 145): yields one rollout → flattens agents → computes
  GAE → runs n_epochs passes, each with frames_per_batch / batch_size
  mini-batches.
  3. Stops when collected_frames >= total_frames.

  So the current total_frames: 1024 would give you 1 iteration (since 65536 >
  1024), then stop. No chance to see loss trends.
Recommendation for the test

  ┌─────────────────────┬─────────┬──────────┬───────────────────────────────┐
  │        Param        │ Current │ Proposed │              Why              │
  ├─────────────────────┼─────────┼──────────┼───────────────────────────────┤
  │ num_envs            │ 256     │ 256      │ Keep GPU saturated — Isaac    │
  │                     │         │          │ Lab's whole design            │
  ├─────────────────────┼─────────┼──────────┼───────────────────────────────┤
  │                     │         │          │ 32 steps/env → fast           │
  │ frames_per_batch    │ 65536   │ 8192     │ iteration, still 40,960 agent │
  │                     │         │          │  transitions                  │
  ├─────────────────────┼─────────┼──────────┼───────────────────────────────┤
  │ total_frames        │ 1024    │ 32768    │ 4 iterations → enough to see  │
  │                     │         │          │ reward/loss trends            │
  ├─────────────────────┼─────────┼──────────┼───────────────────────────────┤
  │                     │         │          │ 8192/64 = 128 mini-batches    │
  │ batch_size          │ 128     │ 64       │ per epoch, good gradient      │
  │                     │         │          │ noise                         │
  ├─────────────────────┼─────────┼──────────┼───────────────────────────────┤
  │ n_epochs            │ 10      │ 5        │ Less overfitting risk on      │
  │                     │         │          │ small batches                 │
  ├─────────────────────┼─────────┼──────────┼───────────────────────────────┤
  │ checkpoint_interval │ 100     │ 4        │ Save at end of run            │
  └─────────────────────┴─────────┴──────────┴───────────────────────────────┘

  This gives you 4 iterations × 5 epochs × 128 mini-batches = 2560 gradient 
  steps, collecting 32,768 frames total across 256 parallel envs. You'll see the
   PPO loss and reward curves start to move, and it should finish in a few
  minutes.
and


python scripts/torchrl/mappo_train.py \
    --task FullTask-TorchRL-UAVSwarm-Direct-v0 \
    --headless
