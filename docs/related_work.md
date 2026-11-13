# Related work (search of 2026-11-13)

Verification: R = abstract and body text read; S = web-search snippet only, not opened. Check S entries before citing.

## Lidar / onboard-sensing swarm navigation with learned policies
- R. Choi et al., Communication-Free Collective Navigation for a Swarm of UAVs via LiDAR-Based Deep RL, arXiv 2601.13657. PPO, leader holds waypoints, lidar occupancy grid + 6 nearest neighbours, 5 real UAVs, scales to 10 followers; sim neighbours from ground truth with noise.
- Huang et al., Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep RL, arXiv 2309.13285 (ICRA 2024). IPPO, attention over K=2 neighbours + SDF obstacles, 32 robots in sim, Crazyflie 2.1.
- Huang et al., Decentralized Control of Quadrotor Swarms with End-to-end Deep RL, arXiv 2109.07735 (S). Attention > deep sets > MLP for neighbour encoding.
- Cooperative Multi-UAV Navigation in Complex Environments via Systematic MADRL, arXiv 2607.25754 (S). 12 m lidar, 8 sector minima.
- Cooperative guidance with SAC and shared voxel map, arXiv 2607.25728 (S).
- LEARN, arXiv 2511.17765 (S). Two-stage safety-guided RL, Crazyflie 2.1.
- Nearest-Neighbor-based Collision Avoidance for Quadrotors via RL (BICARL), arXiv 2104.14912 (S).

## Vision / model-based / imitation
- Schilling, Schiano, Floreano, Vision-based Drone Flocking in Outdoor Environments, RA-L 2021, arXiv 2012.01245 (S).
- Soria, Schiano, Floreano, Predictive control of aerial swarms in cluttered environments, Nature Mach. Intell. 2021 (S). MPC with potential-field terms, 5 real quadrotors.
- Tolstaya et al., Learning Decentralized Controllers for Robot Swarms with GNNs, CoRL 2019, arXiv 1903.10527 (S). Imitation, multi-hop communication.

## Target enclosing, containment, density control
- Formation Control for Moving Target Enclosing via Relative Localization, arXiv 2307.15510 (S); follow-up 2410.14407 (S).
- Safe Decentralized Density Control of Multi-Robot Systems (PDE-constrained), arXiv 2510.20643 (S).
- Decentralized UAV Swarms for Ground Target Protection in GPS- and Communication-Denied Environments, arXiv 2607.20710 (S).
- Distributed Stochastic Optimal Control for Pattern-Oriented Swarms, arXiv 2609.12959 (S).
- Curriculum-learning MARL for cooperative encirclement; extrinsic-and-intrinsic reward MARL for UAV target encirclement (NWPU) (S).

## Hybrid learned / classical, residual
- SPA-MARL, Sub-optimal Policy Aided MARL for Flocking Control (APF prior) (S).
- Calibrate Once, Fly Any Team, arXiv 2609.17265 (S). Residual dynamics correction, not a policy residual.
- No paper found with a residual RL policy on top of an APF swarm controller (not exhaustive).

## Scaling across swarm size
- LEGO-MAPPO equivariant GNN, arXiv 2509.14431 (S). Crazyflie experiment.
- Symmetries-enhanced MARL, arXiv 2501.01136 (S). Quadrotor swarms, varying sizes.
- Network distributed MARL for consensus of quadcopters, arXiv 2606.02107 (S).
- pH-MARL, arXiv 2401.00212 (S). 4x larger teams at deployment.
- CAMAR benchmark (AAAI) (S).

## Sensing hardware and relative localization
- Li et al., onboard range-based relative localization for Crazyflie swarm, arXiv 2003.05853 (S).
- Mr. Virgil: learned visual-range relative localization, arXiv 2512.10540 (S).
- Multi-target pursuit by a heterogeneous Crazyflie swarm with MADRL, arXiv 2303.01799 (S).
- Swarming out of the Lab: comparison of relative localization methods (S).

## Surveys
- A Survey on UAV Control with Multi-Agent RL (S); Scalable and Cooperative DRL Approaches for Multi-UAV Systems: A Systematic Review (S); Multi-UAV RL With Realistic Communication Models (S); review of RL for multirotor UAVs by control layer, Drones 2026 (S).
