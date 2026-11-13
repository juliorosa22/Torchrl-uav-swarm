Moving-sphere packing, curved planar path (amp 1 m, chord 8 m), 128 episodes per speed, spacing 0.5 m, success = no collision and packed >= 0.9.
Gains (kp, repel, kd, ff, rscale) from ES on packed fraction minus 2*collision rate.
- oracle5: 2.39 1.32 1.49 0.81 1.28 (full-state sensing)
- lidar5: 0.984 1.876 3.441 1.323 0.986 (82-beam lidar, no communication)
- oracle20: 2.283 1.342 1.640 1.050 1.735
- lidar20: 0.688 1.072 2.339 1.020 1.892
Files: <name>_final.jsonl, one row per episode. Reproduce: scripts/torchrl/apf_tune.py --final_params ... [--lidar] [--num_agents N].
