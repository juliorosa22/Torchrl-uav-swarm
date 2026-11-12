# APF stress test (PackingMono, kp=2 repel=2 kd=1, 128 episodes per row, 64 envs)

Run: `apf_tune.py --sweep` (sensing/time limit, N=5), `--grid --num_agents N` (swarm size), `--diag` (geometry at step 900).
Rows in swarm_size_clean_N3_8_10_15.jsonl are in order N=3, 8, 10, 15.

- N=5 goal reached: clean 99%; pos noise <=0.2 m ~98-100%, 0.3 m 0%; vel noise <=0.6 m/s >=98%; dropout 0.3 100%, 0.6 0%, 0.9 0% (99% collisions);
  delay 2/5/10 steps 100/72/30% (collisions 0/28/70%); time limit 700/500/350 steps 70/38/8%.
- Swarm size: N=3 100%, N=8/10/15 0%.
- N>=8 failure is not the containment sphere: all drones are inside it, enlarging it (packing_density 0.2) changes nothing, and nearest-neighbour spacing settles at ~0.45-0.78 m vs the 0.9 m requirement. Geometry is feasible (spacing >=0.945 m fits for N<=15).
