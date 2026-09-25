# Decentralized collision avoidance with RL under partial observability: literature notes

Compiled 2026-09-21 from a web search session. Question that prompted it: can RL alone, with only a
partial observation of the nearest neighbor(s), give reliable inter-agent collision avoidance, or do we
need a planning / safety component on top?

**Reading-status legend** (be careful which claims are verified):

- **[FULL]** I extracted and read the paper's full text; numbers below are quoted from it.
- **[ABS]** Only the abstract / search-result snippet was read; treat details as unverified.
- **[LISTED]** Surfaced by the searches but not read at all.

---

## 1. Bottom line

1. The literature does **not** say partial observation of 1-2 neighbors is a blocker. Three quadrotor papers get good decentralized collision avoidance from end-to-end RL with exactly that kind of observation (section 2).
2. Where they differ from our setup is mostly **reward and training design**, not the observation (section 6).
3. When a planner appears, it is rarely a "route planning step" in free space. It is used to (a) generate demonstrations at training time (GLAS, PRIMAL), or (b) break deadlocks in cluttered maps (hybrid RL-MAPF). When guarantees are wanted, the usual add-on is a **safety layer** (control barrier function, velocity obstacle / ORCA) rather than a planner.

---

## 2. RL alone with partial observation

### 2.1 BICARL: Nearest-Neighbor-based Collision Avoidance for Quadrotors via RL [FULL]

Ourari, Cui, Elshamanhory, Koeppl. arXiv:2104.14912 (v3, 2022). <https://arxiv.org/abs/2104.14912>

- **Observation:** only the **single nearest neighbor (k=1)**: distance, bearing, relative velocity. Plus own state and target. Feed-forward, **memoryless** policy (no LSTM / deep sets).
- **Dynamics:** "modified double integrator" with momentum and yaw; the authors state nearest-neighbor info suffices *"in our sufficiently rich motion model"*. Applied to real Crazyflies together with a standard PID controller.
- **Training:** curriculum on swarm size (start with 4 agents on a circle, grow to 40). ~30 min on two desktop CPUs (~1.4e6 steps). Collision defined as distance < 1.5 m, with a penalty radius C = 7 m tuned during training (penalty starts well before the collision distance).
- **Results vs ORCA and FMP:** in package delivery, BICARL collects 15.25 packages/drone with 4 agents down to 4.14 with 14, while ORCA falls to 0 from 10 agents and FMP to 0.25 at 14 (both deadlock). In circle formation change (5 to 50 agents) extra time-to-goal is 4-10 s vs 14-81 s for ORCA; min inter-agent distance shrinks with swarm size (12.4 m at 5 agents, 1.67 m at 50, close to the 1.5 m collision threshold).
- **Caveats:** mostly planar; large length scales; the 3D case is shown only qualitatively.

### 2.2 Batra et al.: Decentralized Control of Quadrotor Swarms with End-to-end Deep RL (CoRL 2021) [FULL]

arXiv:2109.07735. <https://arxiv.org/abs/2109.07735> · project page: <https://sites.google.com/view/swarm-rl>

- **Observation:** own state plus the **relative position and relative velocity of K neighbors** (K = 6, N = 8 quadrotors; K << N for larger swarms).
- **Neighbor encoders:** deep sets (mean of neighbor embeddings) vs attention.
- **Ablation:** a *blind* policy (no neighbor encoder) gets near its target but cannot avoid collisions; a *plain MLP over concatenated neighbor observations* "fails to avoid collisions in most scenarios"; attention is best. The **same-goal scenario** (all goals coincide, r = 0) is described as demanding "very dense configurations with high probabilities of collisions", and it is where attention vs deep sets differs most (deep sets sacrifice formation density to avoid collisions).
- **Collision reward:** `r_col = -alpha_col * 1[new collision] - alpha_prox * sum_j max(1 - dist_ij / d_prox, 0)` where `d_prox` = double the quadrotor frame size. So: a per-collision penalty **plus a smooth proximity penalty summed over all K neighbors**.
- **Collisions are simulated, not terminal:** on contact, a brief random force/torque is applied to both drones with opposite signs; episodes **continue** and drones learn to recover.
- **Vs a classical baseline** (buffered Voronoi cells + PID): classical trajectories are longer, essentially planar, and slower (max 1 m/s vs 4 m/s for the neural controller).
- **Hardware:** Crazyflie 2.0; tiny policies (~1000 parameters, 16/8-neuron encoders) still worked.
- Training scenarios include head-on "swarm-vs-swarm" swaps and dynamic formations.

### 2.3 Huang et al.: Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep RL (ICRA 2024) [FULL]

arXiv:2309.13285. <https://arxiv.org/abs/2309.13285> · code: <https://github.com/Zhehui-Huang/quad-swarm-rl> · project page: <https://sites.google.com/view/obst-avoid-swarm-rl>

- **Neighbors sensed (32 robots):** K = 1 works, **K = 2 is best**, K = 31 is worse ("increase in the input dimension, increasing the hardness of the learning problem").
- **Pre-collision replay buffer:** when a collision is detected, store the environment state **1.5 s before** it; with probability alpha_r, start an episode from a stored state instead of a fresh one; drop a stored state once it has been replayed more than a threshold (assumed too hard). Motivation: a collision lasts 1-2 steps out of a 1500-step episode, so the reward is sparse. **Ablation: collision rate 0.05 -> 0.12 without it.** It also beats prioritized level replay by ~5% success.
- **Attention over neighbors and obstacles:** removing multi-head attention drops success 0.88 -> 0.79.
- **Baselines (8 robots, 20% obstacle density):** GLAS 0.75 success / 0.25 collision, 15 ms inference; SBC (classical safety barrier certificates) 0.99 / 0.01, 21 ms; ours 0.97 / 0.03, **5 ms**. Training from scratch (0.88 / 0.04) beats policy distillation (0.72 / 0.28).
- Scales to 32 robots in simulation with high obstacle density; real-world tests with 8 physical quadrotors.

---

## 3. Variable neighbors and velocity-obstacle-shaped RL [ABS]

- **RL-RVO**: Han et al., "Reinforcement Learned Distributed Multi-Robot Navigation with Reciprocal Velocity Obstacle Shaped Rewards", RA-L 2022. <https://arxiv.org/abs/2203.10229> · code <https://github.com/hanruihua/rl_rvo_nav>. Encodes (reciprocal) velocity-obstacle vectors as observations, a bidirectional recurrent module handles a varying number of neighbors, and the reward uses the RVO area and **expected collision time**.
- **GA3C-CADRL** (Everett, Chen, How, IROS 2018): LSTM over a variable number of agents, formulated as a partially observable problem. Search by title; no verified link. Related implementation (CADRL): <https://github.com/ChanganVR/CADRL>.
- [LISTED] Long et al., "Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep RL": <https://arxiv.org/abs/1709.10082>; "Fully Distributed Multi-Robot Collision Avoidance via Deep RL...": <https://arxiv.org/pdf/1808.03841>; "Multi-agent Motion Planning for Dense and Dynamic Environments via Deep RL": <https://arxiv.org/pdf/2001.06627>.

## 4. Planner plus decentralized execution [ABS]

- **GLAS**: Rivière, Hönig, Yue, Chung, RA-L 2020. <https://arxiv.org/abs/2002.11807> · code <https://github.com/bpriviere/glas>. (i) A **global planner generates demonstrations offline** and local observations are extracted; (ii) deep **imitation learning** of a decentralized policy needing only relative state of nearby neighbors/obstacles; (iii) a **differentiable safety module** for collision-free operation, allowing end-to-end training. Reported ~20% higher success than ORCA; run on an aerial swarm on microcontrollers.
- **PRIMAL**: Sartoretti et al., RA-L 2019. <https://arxiv.org/abs/1809.03531>. RL (A3C) plus behavior cloning from a centralized MAPF expert; fully decentralized, partially observable; up to 1024 agents. Follow-ups [LISTED]: PRIMAL2 <https://arxiv.org/abs/2010.08184>, DHC <https://arxiv.org/pdf/2106.11365>.
- **Deadlock-Free Hybrid RL-MAPF Framework for Zero-Shot Multi-Robot Navigation**. <https://arxiv.org/abs/2511.22685>. RL does reactive avoidance; a safety layer monitors progress and, on detecting a deadlock, triggers a MAPF planner to build feasible trajectories and regulate waypoint progression. (Abstract does not say whether execution is decentralized.)
- [LISTED] Belief-driven hybrid RL under partial observability: <https://www.mdpi.com/2218-6581/15/8/143>; RL + trajectory optimization: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12190238/>; hybrid motion planning with DRL: <https://arxiv.org/html/2512.24651v1>; V-RVO: <https://arxiv.org/pdf/2102.13281>; DMCA (attention + communication): <https://arxiv.org/pdf/2209.06415>.

## 5. Safety layers and classical decentralized planning

- **GCBF+** [ABS]: Zhang, So, Garg, Fan, T-RO 2025. <https://arxiv.org/abs/2401.14554> · code <https://github.com/MIT-REALM/gcbfplus>. Graph neural network parameterizes a control barrier function and a distributed policy; safety "using only local information"; takes LiDAR point clouds; hardware on a Crazyflie swarm; reported to beat leading RL methods by up to 40% at 1024 agents without trading goal-reaching for safety.
- **SBC** (safety barrier certificates): the classical safety-filter baseline in Huang et al. (0.99 / 0.01, but ~4x slower inference than the learned policy).
- **Decentralized MPC / trajectory planning** [ABS]: Luis, Vukosavljev, Schoellig, "Online Trajectory Generation with Distributed MPC for Multi-Robot Motion Planning", RA-L 2020 (Crazyflie-modeled agents). <https://arxiv.org/pdf/1909.05150v1>. DCAD (ORCA planes as constraints inside a flat-MPC, downwash-aware): <https://arxiv.org/pdf/1909.03961>. Common pattern: neighbor trajectories are treated as moving obstacles, so agents need to predict or share them. [LISTED] DMPC-Swarm on nano UAVs: <https://link.springer.com/article/10.1007/s10514-025-10211-w>.
- **Buffered Voronoi cells** and **ORCA** are the classical baselines cited by Batra et al. and BICARL (ref. [10] and [7] respectively in those papers).
- **AttentionSwarm** [FULL, NOT a good fit]: <https://arxiv.org/abs/2503.07376>. Read in full: it is a **centralized** MAPPO-Lagrangian with a CBF attention network, run on gym-pybullet-drones with Vicon localization, **not** a decentralized partial-observation design. Its reward is `1/(d+eps)` if no collision and **-100 on collision** (positive shaped reward plus a large collision penalty); reports 95-100% collision-free navigation. (An automated summary of this paper I first got claimed decentralized execution; the full text contradicts that.)
- [LISTED] SafeSwarm: <https://arxiv.org/pdf/2501.07566>; Neural-Swarm: <https://arxiv.org/pdf/2003.02992>; cooperative collision avoidance of UAV swarms leveraging domain knowledge: <https://arxiv.org/pdf/2507.10913>; communication-free LiDAR-based collective navigation: <https://arxiv.org/pdf/2601.13657>; QuadSwarm simulator: <https://arxiv.org/html/2306.09537>.

## 6. Reward / termination bias (relevant to our reward)

- Termination bias: with negative per-step rewards, ending the episode early can look attractive; the standard mitigations are a **termination penalty** or an **alive bonus**. Source that states this: <https://arxiv.org/html/2009.09467> (adversarial imitation learning context, but the mechanism is general). [LISTED] "Reinforcement Learning with a Terminator": <https://arxiv.org/pdf/2205.15376>.

---

## 7. How our setup differs (my inference, NOT established by these papers)

Relevant code: `rewards.py::get_swarm_gravity_rewards`, `termination.py` (`AGENT_COLLISION_DISTANCE = 0.15`), `sensing.py` (nearest neighbor + mean-pooled neighbor features), `controller.py::compute_swarm_gravity_baseline_action` (APF base of the residual policy).

1. **Reward sign + collision termination.** Our per-step reward is `-distance + safety_penalty`, never positive, and an inter-agent collision **terminates** the episode. That is the termination-bias setup. Consistent with (not proof of) our data: stage 10 (long trips) had 60-100% collision terminations; short PackingSwarm episodes had far fewer. Batra et al. do not terminate on collision.
2. **Collision penalty shape.** Ours is quadratic on the **nearest neighbor only** and has no per-collision event term; Batra et al. sum a proximity penalty over all K neighbors and add a per-collision term.
3. **Neighbor encoding in the dense same-goal case.** We give the policy nearest-neighbor features plus a mean-pooled summary through a small MLP (no attention). Batra et al. found attention needed in the same-goal scenario; Huang et al. found K = 2 best.
4. **No pre-collision replay** and no curriculum on swarm density/size (Huang et al.; BICARL).
5. **Residual base is APF with nearest-neighbor repulsion**, the class of reactive method that BICARL/GLAS show can deadlock.

Our observed numbers for context (200k-1M frames, local): stage 8 (approach) plateaus at ~50% success with ~31% collision terminations even at 1M frames; PackingSwarm ~95% success in a checkpoint eval; switching two separately-trained policies gave 79% (8 envs) / 50% (1 env, 10 episodes) packed, with collisions clustering within ~0.4 s after the last agent's handoff.

## 8. Candidate next steps (cheapest first; each is a testable hypothesis)

1. **Termination-bias test:** add a terminal collision penalty and/or an alive/positive shift, or do not terminate on collision; compare collision rate at 200k frames.
2. **Penalty shape:** proximity penalty over the 2 nearest neighbors plus a per-collision term.
3. **Pre-collision replay** (Huang et al.): store states ~1.5 s before collisions and restart episodes from them.
4. **Decentralized safety layer** on the action (control barrier function or velocity-obstacle/ORCA-style filter). Needs only the relative position/velocity we already observe; no planner.
5. **Attention over per-neighbor features** (Batra/Huang) for the dense same-goal convergence.
6. **GLAS/PRIMAL-style planner-generated demonstrations**: only if we later want guarantees or cluttered environments; it is the heaviest option.

## 9. Caveats

- Papers use very different length scales, dynamics and simulators (e.g. BICARL's 1.5 m collision radius / 7 m penalty radius vs our 0.15 m / 1.0 m); numbers are not directly comparable.
- BICARL's nearest-neighbor sufficiency claim is explicitly conditional on a momentum-aware ("sufficiently rich") motion model.
- Only sections 2.1-2.3 (and AttentionSwarm) were read in full; everything tagged [ABS] / [LISTED] should be read before being cited.

## 10. Suggested reading order

1. Batra et al. (2109.07735): closest to our platform and scenario (same-goal), full reward given.
2. Huang et al. (2309.13285): pre-collision replay + K study.
3. BICARL (2104.14912): the nearest-neighbor-only claim and its conditions.
4. GLAS (2002.11807) and GCBF+ (2401.14554): if we decide to add a planner-derived or safety-layer component.
5. RL-RVO (2203.10229): velocity-obstacle-shaped observation/reward, a possible cheap upgrade to our reward.
