"""Black-box tuning of the radius-mode APF controller (kp, repel_gain, damping kd) on the packing task.

One Isaac Sim process evaluates many parameter sets by stepping the env with the APF action only.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="PackingMono-TorchRL-UAVSwarm-Direct-v0")
parser.add_argument("--config", type=str, default="scripts/torchrl/torchrl_mappo_cfg_local.yaml")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=128, help="Completed episodes per candidate.")
parser.add_argument("--warmup_steps", type=int, default=1100, help="Discarded steps after reset (startup staggering).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--iterations", type=int, default=8)
parser.add_argument("--popsize", type=int, default=8)
parser.add_argument("--init", type=float, nargs=3, default=[2.0, 2.0, 1.0], metavar=("KP", "REPEL", "KD"))
parser.add_argument("--init_sigma", type=float, default=0.4, help="Initial log-space std.")
parser.add_argument("--out", type=str, default="apf_tune.jsonl")
parser.add_argument("--grid", action="store_true", help="Evaluate --init only, once, and exit.")
parser.add_argument("--stage", type=int, default=None)
parser.add_argument("--num_agents", type=int, default=None)
parser.add_argument("--diag", action="store_true", help="Print the pre-timeout swarm geometry for several gain sets.")
parser.add_argument("--packing_density", type=float, default=None)
parser.add_argument("--move", action="store_true", help="Moving-sphere task: sweep sphere speed with/without velocity feed-forward.")
parser.add_argument("--move_speeds", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5])
parser.add_argument("--sep", type=float, default=0.5, help="Relaxed spacing threshold (m) for the packed state.")
parser.add_argument("--tune_move", action="store_true", help="ES over (kp, repel, kd, ff, radius_scale) on packed fraction in the moving task.")
parser.add_argument("--final_params", type=float, nargs=5, default=None, metavar=("KP", "REPEL", "KD", "FF", "RSCALE"), help="Evaluate one tuned set with --move_stats-style rows.")
parser.add_argument("--tune_speeds", type=float, nargs="+", default=[0.5, 1.0, 1.5])
parser.add_argument("--move_curve_amp", type=float, default=0.0)
parser.add_argument("--move_planar", action="store_true")
parser.add_argument("--move_paths", type=str, nargs="+", default=None, help="Path families (paths.py): line curve slalom arc hairpin spline figure8 coaster helix; 'train' / 'held' / 'all' shortcuts.")
parser.add_argument("--move_profiles", type=str, nargs="+", default=["const"], help="Speed profiles: const ramp stopgo.")
parser.add_argument("--move_path_len", type=float, default=10.0)
parser.add_argument("--move_speed_range", type=float, nargs=2, default=None, metavar=("LO", "HI"), help="Per-episode speed range (m/s) in family mode.")
parser.add_argument("--move_path_lens", type=float, nargs="+", default=None, help="Per-episode path-length buckets (m) in family mode.")
parser.add_argument("--move_a_max", type=float, default=3.0, help="Path acceleration budget (m/s^2): curvature floor is speed^2 / a_max.")
parser.add_argument("--move_z_max", type=float, default=4.0, help="Top of the sphere-centre altitude band in family mode (m).")
parser.add_argument("--move_helix", type=float, nargs=3, default=None, metavar=("RADIUS", "CLIMB", "TURNS"), help="Helical sphere path instead of the chord.")
parser.add_argument("--move_distance", type=float, default=6.0)
parser.add_argument("--sense_pos", type=float, nargs=2, default=[0.0, 0.0], metavar=("A", "B"), help="Neighbour position noise sigma = A + B*distance (m).")
parser.add_argument("--sense_vel", type=float, default=0.0, help="Neighbour relative-velocity noise sigma (m/s).")
parser.add_argument("--sense_drop", type=float, nargs=2, default=[0.0, 0.0], metavar=("A", "B"), help="Link-loss probability = A + B*distance.")
parser.add_argument("--sense_range", type=float, default=0.0, help="Sensing range (m); 0 = unlimited.")
parser.add_argument("--sense_hz", type=float, default=0.0, help="Neighbour measurement rate (Hz); 0 = every control step.")
parser.add_argument("--sense_delay", type=int, default=0, help="Neighbour measurement latency (control steps).")
parser.add_argument("--lidar", action="store_true", help="Neighbours seen only through an onboard lidar scan (no communication).")
parser.add_argument("--lidar_az", type=int, default=16)
parser.add_argument("--lidar_elev", type=float, nargs="+", default=[-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0], help="Ring elevations (deg); +-90 are single up/down beams.")
parser.add_argument("--lidar_range", type=float, default=3.0)
parser.add_argument("--lidar_radius", type=float, default=0.1, help="Drone body radius seen by the rays (m).")
parser.add_argument("--lidar_noise", type=float, nargs=2, default=[0.02, 0.01], metavar=("A", "B"))
parser.add_argument("--lidar_miss", type=float, default=0.02)
parser.add_argument("--lidar_hz", type=float, default=10.0)
parser.add_argument("--lidar_delay", type=int, default=1)
parser.add_argument("--lidar_check", action="store_true", help="Ray-hit sanity check against ground truth.")
parser.add_argument("--policy_ckpt", type=str, default=None, help="Evaluate a trained residual policy (moving-task obs) on top of the APF given by --final_params.")
parser.add_argument("--policy_arch", type=str, default="attention", choices=["attention", "flat"])
parser.add_argument("--obs_sense", type=str, default="oracle", choices=["oracle", "lidar"])
parser.add_argument("--obs_k", type=int, default=8)
parser.add_argument("--residual_scale", type=float, default=0.5)
parser.add_argument("--zero_policy", action="store_true", help="With --policy_ckpt-style obs: apply no residual (pure APF through the moving-obs env).")
parser.add_argument("--kv", type=float, default=None, help="Overrides controller.Kv (outer-loop velocity P gain).")
parser.add_argument("--preview", action="store_true", help="Tune a preview APF: attraction toward the sphere centre tau_p ahead and feed-forward from its mean velocity over tau_f (adds 2 ES parameters).")
parser.add_argument("--preview_params", type=float, nargs=2, default=None, metavar=("TAU_P", "TAU_F"), help="With --final_params: evaluate the preview APF with these look-ahead times (s).")
parser.add_argument("--max_vel", type=float, default=None, help="Overrides controller.max_lin_vel_cmd (m/s per axis).")
parser.add_argument("--gain_mod", type=float, default=0.0, help="Evaluate a gain-modulation policy (5 log-multipliers in [-1,1]*this on the final APF gains).")
parser.add_argument("--wp_horizon", type=int, default=0, help="Evaluate a waypoint-plan policy (K body-frame increments per drone).")
parser.add_argument("--wp_dt", type=float, default=0.2)
parser.add_argument("--wp_lookahead", type=float, default=0.3)
parser.add_argument("--wp_goal_bias", action="store_true")
parser.add_argument("--residual_kp", type=float, default=2.0, help="Waypoint mode: goal-bias attraction gain.")
parser.add_argument("--move_stats", action="store_true", help="Per-episode inter-agent distance statistics on the moving-sphere task.")
parser.add_argument("--sweep", action="store_true", help="Stress-test the --init gains over sensing/time-limit conditions.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

import sys  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import UavSwarm.tasks  # noqa: F401, E402
from UavSwarm.tasks.direct.torchrl_swarm.controller import (  # noqa: E402
    apf_sense_update,
    compute_apf_gain_mod_action,
    compute_goal_bias,
    compute_swarm_gravity_baseline_action,
    compute_waypoint_plan_action,
)
from UavSwarm.tasks.direct.torchrl_swarm import paths as _pf  # noqa: E402
from UavSwarm.tasks.direct.torchrl_swarm.lidar import lidar_update, make_lidar  # noqa: E402

LOG_LO = torch.log(torch.tensor([0.3, 0.2, 0.05]))
LOG_HI = torch.log(torch.tensor([8.0, 16.0, 8.0]))


class Evaluator:
    def __init__(self):
        with open(args_cli.config) as f:
            config = yaml.safe_load(f)
        cfg = parse_env_cfg(args_cli.task, device=config["env"]["device"], num_envs=args_cli.num_envs)
        for k, v in config.get("controller", {}).items():
            setattr(cfg.controller, k, v)
        cfg.controller.type = "geometric"
        if args_cli.max_vel is not None:
            cfg.controller.max_lin_vel_cmd = args_cli.max_vel
        if args_cli.kv is not None:
            cfg.controller.Kv = args_cli.kv
        cfg.seed = args_cli.seed
        if args_cli.stage is not None:
            cfg.curriculum.active_stage = args_cli.stage
        n = args_cli.num_agents
        if n is not None and n != cfg.num_agents:
            cfg.num_agents = n
            cfg.possible_agents = [f"robot_{i}" for i in range(n)]
            cfg.action_spaces = {f"robot_{i}": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for i in range(n)}
            cfg.observation_spaces = {
                f"robot_{i}": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(cfg.single_observation_space,))
                for i in range(n)
            }
            cfg.state_space = n * cfg.single_observation_space
        if args_cli.packing_density is not None:
            cfg.curriculum.stage8_packing_density = args_cli.packing_density
        self.net = None
        self._obs = None
        self._wp_prev = None
        if args_cli.policy_ckpt or args_cli.zero_policy:
            from UavSwarm.tasks.direct.torchrl_swarm.moving_task import configure_cfg

            lidar_cfg = dict(
                n_az=args_cli.lidar_az, elev_deg=tuple(args_cli.lidar_elev), rng_max=args_cli.lidar_range, radius=args_cli.lidar_radius,
                sig_a=args_cli.lidar_noise[0], sig_b=args_cli.lidar_noise[1], miss=args_cli.lidar_miss,
                period=max(1, round(1.0 / (args_cli.lidar_hz * cfg.sim.dt * cfg.decimation))), delay=args_cli.lidar_delay,
            )
            configure_cfg(cfg, args_cli.obs_sense, args_cli.obs_k, lidar_cfg=lidar_cfg)
        self.env = gym.make(args_cli.task, cfg=cfg)
        self.u = self.env.unwrapped
        if args_cli.policy_ckpt:
            self._load_policy()
        self.names = [f"robot_{i}" for i in range(self.u.num_drones)]
        self.default_len = self.u.max_episode_length
        self.step_dt = self.u.cfg.sim.dt * self.u.cfg.decimation
        print(f"[tune] radius mode={self.u.cfg.swarm_cfg.neighborhood_mode} stage={self.u.cfg.curriculum.active_stage}")

    def _load_policy(self):
        from mappo_torchl import AttentionMAPPOPolicy, MAPPOPolicy

        sd = torch.load(args_cli.policy_ckpt, map_location="cpu")["policy"]
        sd = {k[len("module.0.module."):]: v for k, v in sd.items() if k.startswith("module.0.module.")}
        d = self.u.cfg.single_observation_space
        if args_cli.wp_horizon > 0 and args_cli.policy_arch == "flat":
            from mappo_torchl import MAPPOPolicy as _Flat

            with open(args_cli.config) as f:
                hs = yaml.safe_load(f)["models"]["policy"]["hidden_sizes"]
            self.net = _Flat(d + 3 * args_cli.wp_horizon, 3 * args_cli.wp_horizon, hs)
        elif args_cli.wp_horizon > 0:
            k = self.u.cfg.moving_obs_k
            tdim = 5 if args_cli.obs_sense == "lidar" else 6
            self.net = AttentionMAPPOPolicy(d - k * tdim, k, 3 * args_cli.wp_horizon, token_dim=tdim, masked=tdim == 5,
                                            extra_dim=3 * args_cli.wp_horizon)
        elif args_cli.policy_arch == "attention":
            k = self.u.cfg.moving_obs_k
            tdim = 5 if args_cli.obs_sense == "lidar" else 6
            self.net = AttentionMAPPOPolicy(d - k * tdim, k, 5 if args_cli.gain_mod > 0 else 4, token_dim=tdim, masked=tdim == 5)
        else:
            self.net = MAPPOPolicy(d, 4, (256, 256, 128))
        self.net.load_state_dict(sd)
        self.net.to(self.u.device).eval()

    def _policy_action(self):
        obs = torch.stack([self._obs[n] for n in self.names], dim=1).to(self.u.device)
        mean, _ = self.net(obs)
        return mean.clamp(-1.0, 1.0) if args_cli.gain_mod > 0 else torch.tanh(mean)

    def _wp_step(self, kp, repel):
        a, u = args_cli, self.u
        K = a.wp_horizon
        if self._wp_prev is None:
            self._wp_prev = torch.zeros(u.num_envs, u.num_drones, 3 * K, device=u.device)
        if self.net is not None:
            obs = torch.cat([torch.stack([self._obs[n] for n in self.names], dim=1).to(u.device), self._wp_prev], dim=-1)
            raw = torch.tanh(self.net(obs)[0])
        else:
            raw = torch.zeros_like(self._wp_prev)
        step_max = u.cfg.controller.max_lin_vel_cmd * a.wp_dt
        incr = raw.reshape(u.num_envs, u.num_drones, K, 3)
        if a.wp_goal_bias:
            incr = (incr + compute_goal_bias(u, a.residual_kp).unsqueeze(2)).clamp(-1.0, 1.0)
        act = compute_waypoint_plan_action(u, (step_max * incr).cumsum(dim=2), a.wp_dt, a.wp_lookahead, True, repel)
        self._obs, _, _, _, _ = self.env.step({n: act[:, i] for i, n in enumerate(self.names)})
        self._wp_prev = raw.detach().clone()
        self._wp_prev[(u._last_terminated | u._last_timed_out)] = 0.0

    def _step(self, kp, repel, ff=0.0):
        if args_cli.wp_horizon > 0:
            return self._wp_step(kp, repel)
        if self.net is not None or args_cli.zero_policy:
            if self.net is not None and args_cli.gain_mod > 0:
                u = self.u
                base = (kp, repel, u.cfg.swarm_cfg.apf_damping_gain, ff, getattr(u, "_apf_radius_scale", 1.0))
                act = compute_apf_gain_mod_action(u, torch.exp(args_cli.gain_mod * self._policy_action()), base)
                self._obs, _, _, _, _ = self.env.step({n: act[:, i] for i, n in enumerate(self.names)})
                return None
            act = compute_swarm_gravity_baseline_action(self.u, kp, repel, ff_gain=ff)
            if self.net is not None:
                act = (act + args_cli.residual_scale * self._policy_action()).clamp(-1.0, 1.0)
            self._obs, _, _, _, _ = self.env.step({n: act[:, i] for i, n in enumerate(self.names)})
            return None
        if getattr(self.u, "_lidar", None) is not None:
            if not self.u._lidar.get("auto"):
                lidar_update(self.u)
        elif self.u._apf_sense is not None:
            apf_sense_update(self.u)
        act = compute_swarm_gravity_baseline_action(self.u, kp, repel, ff_gain=ff)
        return self.env.step({n: act[:, i] for i, n in enumerate(self.names)})

    def _track(self, acc):
        u = self.u
        moving = u._target_left > 0
        if not bool(moving.any()):
            return
        pos = torch.stack([r.data.root_pos_w for r in u._robots], dim=0)
        dist = (u._shared_target_w.unsqueeze(0) - pos).norm(dim=2)
        radius = u.cfg.curriculum.get_containment_radius(u.num_drones, u.cfg.swarm_cfg.min_safe_distance)
        pair = (pos.unsqueeze(1) - pos.unsqueeze(0)).norm(dim=-1) + 10 * torch.eye(u.num_drones, device=u.device).unsqueeze(-1)
        spaced = (pair.min(dim=1).values >= u.cfg.reward_cfg.pack_min_sep_frac * u.cfg.swarm_cfg.min_safe_distance).all(0)
        acc["moving"] += int(moving.sum())
        acc["inside"] += int(((dist <= radius).all(0) & moving).sum())
        acc["spaced"] += int((spaced & moving).sum())
        acc["maxd"] = max(acc["maxd"], float(dist.max(0).values[moving].max()))

    @torch.no_grad()
    def evaluate_stats(self, kp, repel, kd, move_speed, ff, episodes, tag):
        """Moving-sphere episode report. No early goal termination: an episode runs the full path
        plus the hold, ending on timeout or collision. A step is 'packed' if every drone is inside the
        sphere and the nearest-neighbour distance is >= --sep."""
        u, dev = self.u, self.u.device
        E, D = u.num_envs, u.num_drones
        u.cfg.swarm_cfg.apf_damping_gain = kd
        a = args_cli
        policy_mode = self.net is not None or a.zero_policy or a.wp_horizon > 0
        period = max(1, round(1.0 / (a.sense_hz * self.step_dt))) if a.sense_hz > 0 else 1
        realistic = a.sense_pos != [0.0, 0.0] or a.sense_vel > 0 or a.sense_drop != [0.0, 0.0] or a.sense_range > 0 or period > 1 or a.sense_delay > 0
        if policy_mode:
            u._apf_sense = None
        elif a.lidar:
            u._lidar = None
            u._apf_sense = None
            u._lidar = make_lidar(a.lidar_az, a.lidar_elev, a.lidar_range, 0.1, a.lidar_radius, a.lidar_noise[0], a.lidar_noise[1],
                                  a.lidar_miss, 0.5, max(1, round(1.0 / (a.lidar_hz * self.step_dt))), a.lidar_delay)
        else:
            u._lidar = None
            u._apf_sense = dict(
                pos_noise=a.sense_pos[0], pos_noise_slope=a.sense_pos[1], vel_noise=a.sense_vel, dropout=a.sense_drop[0],
                dropout_slope=a.sense_drop[1], range=a.sense_range, period=period, delay=a.sense_delay, buf=[],
            ) if realistic else None
        cur = u.cfg.curriculum
        cur.move_target_speed = move_speed
        u.cfg.episode_length_s = cur.moving_episode_s() if not cur.move_paths else (cur.move_path_len / move_speed + cur.move_pause_budget_s + cur.move_settle_s + 0.5)
        radius = cur.get_containment_radius(D, u.cfg.swarm_cfg.min_safe_distance)
        sep = args_cli.sep
        names = ["n_mv", "n_st", "pk_mv", "pk_st", "ins_mv", "ins_st", "sum_nn_mv", "sum_nn_st", "near03", "below_sep",
                 "jit_mv", "jit_st", "spd_mv", "spd_st", "cerr_mv", "cerr_st", "clag_mv", "clag_max", "maxd_mv"]
        acc = {}

        def reset_acc(mask=None):
            for k in names + ["min_nn"]:
                fill = 99.0 if k == "min_nn" else 0.0
                if mask is None:
                    acc[k] = torch.full((E,), fill, device=dev)
                else:
                    acc[k][mask] = fill

        reset_acc()
        self._obs, _ = self.env.reset()
        self._wp_prev = None
        u.episode_length_buf[:] = 0
        rows, step = [], 0
        while len(rows) < episodes and step < max(12000, int(6 * u.cfg.episode_length_s / self.step_dt)):
            fam_prev, prof_prev, spd_prev = u._path_fam.clone(), u._path_prof.clone(), u._path_speed.clone()
            self._step(kp, repel, ff)
            step += 1
            done = u._last_terminated | u._last_timed_out
            alive = ~done
            pos = torch.stack([r.data.root_pos_w for r in u._robots], dim=0)
            vel = torch.stack([r.data.root_lin_vel_w for r in u._robots], dim=0)
            dist = (u._shared_target_w.unsqueeze(0) - pos).norm(dim=2)
            pair = (pos.unsqueeze(1) - pos.unsqueeze(0)).norm(dim=-1) + 10 * torch.eye(D, device=dev).unsqueeze(-1)
            nn = pair.min(dim=1).values.min(dim=0).values
            cen_v = vel.mean(0)
            jit = ((vel - cen_v).pow(2).sum(-1).mean(0)).sqrt()
            spd = vel.norm(dim=2).mean(0)
            cerr = (cen_v - u._target_vel_w).norm(dim=1)
            clag = (pos.mean(0) - u._shared_target_w).norm(dim=1)
            inside = (dist <= radius).all(0)
            packed = (inside & (nn >= sep)).float()
            mv = ((u._target_left > 0) & alive).float()
            st = ((u._target_left <= 0) & alive).float()
            acc["min_nn"] = torch.where(alive, torch.minimum(acc["min_nn"], nn), acc["min_nn"])
            acc["near03"] += (nn < 0.3).float() * alive
            acc["below_sep"] += (nn < sep).float() * alive
            acc["clag_max"] = torch.where(alive, torch.maximum(acc["clag_max"], clag), acc["clag_max"])
            acc["maxd_mv"] = torch.where(mv > 0, torch.maximum(acc["maxd_mv"], dist.max(0).values), acc["maxd_mv"])
            for sfx, m in (("mv", mv), ("st", st)):
                acc["n_" + sfx] += m
                acc["pk_" + sfx] += packed * m
                acc["ins_" + sfx] += inside.float() * m
                acc["sum_nn_" + sfx] += nn * m
                acc["jit_" + sfx] += jit * m
                acc["spd_" + sfx] += spd * m
                acc["cerr_" + sfx] += cerr * m
            acc["clag_mv"] += clag * mv
            if bool(done.any()):
                reasons = u._termination_reasons
                for e in done.nonzero().flatten().tolist():
                    g = lambda k: float(acc[k][e])
                    n_mv, n_st = max(g("n_mv"), 1.0), max(g("n_st"), 1.0)
                    n_all = max(g("n_mv") + g("n_st"), 1.0)
                    coll = bool(reasons["inter_agent_collision"][e] | reasons["collision"][e] | reasons["out_of_bounds"][e])
                    packed_all = (g("pk_mv") + g("pk_st")) / n_all
                    rows.append(dict(
                        tag=tag, v=move_speed, ff=ff, fam=(_pf.ALL_FAMILIES[int(fam_prev[e])] if u.cfg.curriculum.move_paths else "legacy"),
                        prof=(_pf.PROFILES[int(prof_prev[e])] if u.cfg.curriculum.move_paths else "const"), spd=float(spd_prev[e]), n_agents=D, sep=sep, coll=coll, timeout=bool(u._last_timed_out[e]),
                        steps=n_all, packed_all=packed_all, packed_mv=g("pk_mv") / n_mv, packed_st=g("pk_st") / n_st,
                        inside_mv=g("ins_mv") / n_mv, inside_st=g("ins_st") / n_st,
                        min_nn=g("min_nn"), mean_nn_mv=g("sum_nn_mv") / n_mv, mean_nn_st=g("sum_nn_st") / n_st,
                        near03=g("near03") / n_all, below_sep=g("below_sep") / n_all,
                        jit_mv=g("jit_mv") / n_mv, jit_st=g("jit_st") / n_st, spd_mv=g("spd_mv") / n_mv, spd_st=g("spd_st") / n_st,
                        cerr_mv=g("cerr_mv") / n_mv, cerr_st=g("cerr_st") / n_st,
                        clag_mean_mv=g("clag_mv") / n_mv, clag_max=g("clag_max"), maxd_mv=g("maxd_mv"),
                        success=bool(packed_all >= 0.9 and not coll),
                    ))
                reset_acc(done)
        return rows

    @torch.no_grad()
    def evaluate(self, kp, repel, kd, pos_noise=0.0, vel_noise=0.0, dropout=0.0, delay=0, max_steps=None, move_speed=0.0, ff=0.0):
        self.u.cfg.swarm_cfg.apf_damping_gain = kd
        perturbed = pos_noise > 0 or vel_noise > 0 or dropout > 0 or delay > 0
        self.u._apf_sense = (
            dict(pos_noise=pos_noise, vel_noise=vel_noise, dropout=dropout, delay=delay, buf=[]) if perturbed else None
        )
        cur = self.u.cfg.curriculum
        cur.move_target_speed = move_speed
        if move_speed > 0:
            self.u.cfg.episode_length_s = cur.move_target_distance / move_speed + cur.move_settle_s
        else:
            self.u.cfg.episode_length_s = (max_steps or self.default_len) * self.step_dt
        track = dict(moving=0, inside=0, spaced=0, maxd=0.0)
        self.env.reset()
        if move_speed > 0:
            self.u.episode_length_buf[:] = 0
        keys = ["goal_reached", "inter_agent_collision", "collision", "out_of_bounds"]
        counts = {k: 0 for k in keys}
        n_done = n_timeout = 0
        step = 0
        t0 = time.time()
        while n_done < args_cli.episodes:
            self._step(kp, repel, ff)
            step += 1
            if move_speed > 0 and step > args_cli.warmup_steps:
                self._track(track)
            if step <= args_cli.warmup_steps:
                continue
            done = self.u._last_terminated | self.u._last_timed_out
            n = int(done.sum())
            if n == 0:
                continue
            n_done += n
            n_timeout += int(self.u._last_timed_out[done].sum())
            reasons = self.u._termination_reasons
            for k in keys:
                if k in reasons:
                    counts[k] += int(reasons[k][done].sum())
            if step > args_cli.warmup_steps + 6000:
                break
        res = {k: counts[k] / max(n_done, 1) for k in keys}
        if move_speed > 0:
            res.update(track_inside=track['inside'] / max(track['moving'], 1), track_spaced=track['spaced'] / max(track['moving'], 1), track_maxd=track['maxd'])
        res.update(timeout=n_timeout / max(n_done, 1), n=n_done, steps=step, secs=round(time.time() - t0, 1))
        return res


@torch.no_grad()
def lidar_check(ev, steps=400):
    """Noise-free scan vs ground truth: hit consistency and nearest-neighbour detection rate by distance."""
    u, a = ev.u, args_cli
    u._apf_sense = None
    u.cfg.swarm_cfg.apf_damping_gain = 1.49
    u._apf_radius_scale = 1.28
    cur = u.cfg.curriculum
    cur.move_target_speed, cur.move_planar, cur.move_target_distance = 0.5, True, 8.0
    u.cfg.episode_length_s = 10_000
    u._lidar = make_lidar(a.lidar_az, a.lidar_elev, a.lidar_range, 0.1, a.lidar_radius, 0.0, 0.0, 0.0, 0.0, 1, 0)
    ev.env.reset()
    u.episode_length_buf[:] = 0
    edges = [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 9.9]
    det = torch.zeros(len(edges) - 1)
    cnt = torch.zeros(len(edges) - 1)
    bad = tot = 0
    vis = n_pairs = 0
    el_det, el_cnt = torch.zeros(3), torch.zeros(3)
    for _ in range(steps):
        ev._step(2.39, 1.32, 0.81)
        lidar_update(u)
        rng, dirs = u._lidar["scan"]
        pos = torch.stack([r.data.root_pos_w for r in u._robots], dim=0)
        D, E, B = rng.shape
        hit = rng.isfinite()
        hp = pos.unsqueeze(2) + dirs * rng.nan_to_num(posinf=0.0).unsqueeze(-1)
        dm = torch.cdist(hp.permute(1, 0, 2, 3).reshape(E, D * B, 3), pos.permute(1, 0, 2)).reshape(E, D, B, D).permute(1, 0, 2, 3)
        near = dm.min(-1).values
        bad += int(((near > a.lidar_radius + 1e-3) & hit).sum())
        tot += int(hit.sum())
        owner = dm.argmin(-1)
        pair = (pos.unsqueeze(1) - pos.unsqueeze(0)).norm(dim=-1) + 10 * torch.eye(D, device=pos.device).unsqueeze(-1)
        nn_d, nn_j = pair.min(dim=1)
        seen = torch.zeros(D, E, D, dtype=torch.bool, device=pos.device)
        seen.scatter_(2, owner, hit)
        seen &= ~torch.eye(D, dtype=torch.bool, device=pos.device).unsqueeze(1)
        got = seen.gather(2, nn_j.unsqueeze(-1)).squeeze(-1)
        nn_vec = pos.permute(1, 0, 2).gather(1, nn_j.t().unsqueeze(-1).expand(-1, -1, 3)).permute(1, 0, 2) - pos
        el_nn = torch.rad2deg(torch.asin((nn_vec[..., 2] / nn_d.clamp(min=1e-6)).clamp(-1, 1))).abs()
        close = nn_d < 0.5
        for k, (lo_e, hi_e) in enumerate(((0, 30), (30, 56), (56, 91))):
            m = close & (el_nn >= lo_e) & (el_nn < hi_e)
            el_det[k] += float(got[m].sum())
            el_cnt[k] += float(m.sum())
        vis += int(seen.sum())
        n_pairs += D * E
        for k in range(len(edges) - 1):
            m = (nn_d >= edges[k]) & (nn_d < edges[k + 1])
            det[k] += float(got[m].sum())
            cnt[k] += float(m.sum())
    print(f"[lidar] hits={tot} inconsistent={bad}  mean neighbours seen per drone={vis / n_pairs:.2f} of {u.num_drones - 1}", flush=True)
    for k, nm in enumerate(("0-30", "30-56", "56-90")):
        print(f"[lidar] close (<0.5 m) nearest neighbour at |elevation| {nm} deg: detected {el_det[k] / max(el_cnt[k], 1):.2f} (n={int(el_cnt[k])})", flush=True)
    for k in range(len(edges) - 1):
        print(f"[lidar] nearest neighbour {edges[k]:.1f}-{edges[k + 1]:.1f} m: detected {det[k] / max(cnt[k], 1):.2f} (n={int(cnt[k])})", flush=True)


@torch.no_grad()
def diagnose(ev, kp, repel, kd, steps=900):
    u = ev.u
    u.cfg.swarm_cfg.apf_damping_gain = kd
    u._apf_sense = None
    u.cfg.episode_length_s = 10_000
    ev.env.reset()
    alive = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)
    for _ in range(steps):
        ev._step(kp, repel)
        alive &= ~(u._last_terminated | u._last_timed_out)
    pos = torch.stack([r.data.root_pos_w for r in u._robots], dim=0)
    vel = torch.stack([r.data.root_lin_vel_w for r in u._robots], dim=0)
    radius = u.cfg.curriculum.get_containment_radius(u.num_drones, u.cfg.swarm_cfg.min_safe_distance)
    dist = (u._shared_target_w.unsqueeze(0) - pos).norm(dim=2)
    pair = (pos.unsqueeze(1) - pos.unsqueeze(0)).norm(dim=-1) + 10 * torch.eye(u.num_drones, device=u.device).unsqueeze(-1)
    nn = pair.min(dim=1).values
    sep = u.cfg.reward_cfg.pack_min_sep_frac * u.cfg.swarm_cfg.min_safe_distance
    inside, spaced, slow = dist <= radius, nn >= sep, vel.norm(dim=2) <= u.cfg.reward_cfg.pack_speed_threshold
    a = alive
    return dict(
        kp=kp, repel=repel, kd=kd, N=u.num_drones, radius=round(radius, 3), alive=int(a.sum()),
        max_dist=round(float(dist.max(0).values[a].mean()), 3), mean_dist=round(float(dist.mean(0)[a].mean()), 3),
        min_nn=round(float(nn.min(0).values[a].mean()), 3), mean_nn=round(float(nn.mean(0)[a].mean()), 3),
        max_speed=round(float(vel.norm(dim=2).max(0).values[a].mean()), 3),
        all_inside=round(float(inside.all(0)[a].float().mean()), 3), all_spaced=round(float(spaced.all(0)[a].float().mean()), 3),
        all_slow=round(float(slow.all(0)[a].float().mean()), 3),
        success=round(float((inside & spaced & slow).all(0)[a].float().mean()), 3),
    )


def fitness(res):
    return res["goal_reached"] - 2.0 * res["inter_agent_collision"] - 2.0 * res["collision"]


def main():
    ev = Evaluator()
    if args_cli.lidar_check:
        lidar_check(ev)
        return
    torch.manual_seed(args_cli.seed)
    mean = torch.log(torch.tensor(args_cli.init))
    sigma = torch.full((3,), args_cli.init_sigma)

    def run(p, tag):
        kp, repel, kd = [float(x) for x in torch.exp(p)]
        res = ev.evaluate(kp, repel, kd)
        row = dict(tag=tag, kp=kp, repel=repel, kd=kd, fit=fitness(res), **res)
        print("[tune]", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        with open(args_cli.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row["fit"]

    if args_cli.grid:
        run(mean, "grid")
        return

    if args_cli.tune_move or args_cli.final_params:
        ev.u.cfg.curriculum.move_target_distance = args_cli.move_distance
        ev.u.cfg.curriculum.move_planar = args_cli.move_planar
        ev.u.cfg.curriculum.move_curve_amp = args_cli.move_curve_amp
        if args_cli.move_paths:
            from UavSwarm.tasks.direct.torchrl_swarm import paths as _paths

            short = {"train": _paths.TRAIN_FAMILIES, "held": _paths.HELD_OUT_FAMILIES, "all": _paths.ALL_FAMILIES}
            fams = tuple(f for name in args_cli.move_paths for f in short.get(name, (name,)))
            c_ = ev.u.cfg.curriculum
            c_.move_paths, c_.move_profiles, c_.move_path_len = fams, tuple(args_cli.move_profiles), args_cli.move_path_len
            c_.move_z_max, c_.move_a_max = args_cli.move_z_max, args_cli.move_a_max
            if args_cli.move_path_lens:
                c_.move_path_lens = tuple(args_cli.move_path_lens)
            if args_cli.move_speed_range:
                c_.move_speed_range = tuple(args_cli.move_speed_range)
                args_cli.move_speeds = args_cli.tune_speeds = [args_cli.move_speed_range[1]]
        if args_cli.move_helix:
            cur_ = ev.u.cfg.curriculum
            cur_.move_helix_radius, cur_.move_helix_climb, cur_.move_helix_turns = args_cli.move_helix

        def run_set(kp, repel, kd, ff, rscale, speeds, episodes, tag, preview=None):
            ev.u._apf_radius_scale = rscale
            ev.u._apf_preview = preview
            out = []
            for v in speeds:
                out += ev.evaluate_stats(kp, repel, kd, v, ff, episodes, f"{tag},v={v}")
            for r in out:
                r.update(kp=kp, repel=repel, kd=kd, ff_gain=ff, rscale=rscale, preview=preview)
            return out

        if args_cli.final_params:
            kp, repel, kd, ff, rscale = args_cli.final_params
            rows = run_set(kp, repel, kd, ff, rscale, args_cli.move_speeds, args_cli.episodes, "tuned",
                           tuple(args_cli.preview_params) if args_cli.preview_params else None)
            with open(args_cli.out, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"[final] {len(rows)} episodes written", flush=True)
            return

        lo = torch.tensor([math.log(0.3), math.log(0.2), math.log(0.05), 0.0, math.log(0.6)])
        hi = torch.tensor([math.log(8.0), math.log(16.0), math.log(8.0), 1.5, math.log(2.0)])
        mean = torch.tensor([math.log(args_cli.init[0]), math.log(args_cli.init[1]), math.log(args_cli.init[2]), 1.0, 0.0])
        sigma = torch.tensor([0.4, 0.4, 0.5, 0.25, 0.2])
        floor = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.03])
        if args_cli.preview:
            lo, hi = torch.cat([lo, torch.tensor([0.0, 0.05])]), torch.cat([hi, torch.tensor([1.5, 1.5])])
            mean, sigma = torch.cat([mean, torch.tensor([0.4, 0.4])]), torch.cat([sigma, torch.tensor([0.25, 0.25])])
            floor = torch.cat([floor, torch.tensor([0.03, 0.03])])
        npar = len(mean)

        def decode(p):
            base = (float(p[0].exp()), float(p[1].exp()), float(p[2].exp()), float(p[3]), float(p[4].exp()))
            return base + ((float(p[5]), float(p[6])) if npar == 7 else ())

        def score(p, tag):
            kp, repel, kd, ff, rscale, *pv = decode(p)
            rows = run_set(kp, repel, kd, ff, rscale, args_cli.tune_speeds, args_cli.episodes, tag, tuple(pv) if pv else None)
            per_v = {}
            for r in rows:
                per_v.setdefault(r["v"], []).append(r)
            fit = sum(sum(x["packed_all"] - 2.0 * x["coll"] for x in g) / len(g) for g in per_v.values()) / len(per_v)
            summ = dict(
                tag=tag, kp=kp, repel=repel, kd=kd, ff=ff, rscale=rscale, tp=(pv[0] if pv else None), tf=(pv[1] if pv else None), fit=fit,
                packed={v: sum(x["packed_all"] for x in g) / len(g) for v, g in per_v.items()},
                coll=sum(x["coll"] for x in rows) / len(rows), jit_mv=sum(x["jit_mv"] for x in rows) / len(rows),
                inside_mv=sum(x["inside_mv"] for x in rows) / len(rows), min_nn=sum(x["min_nn"] for x in rows) / len(rows),
            )
            print("[tune]", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}), flush=True)
            with open(args_cli.out, "a") as f:
                f.write(json.dumps(summ) + "\n")
            return fit

        score(mean, "init")
        best_fit, best = -1e9, mean.clone()
        for it in range(args_cli.iterations):
            pop = torch.max(torch.min(mean + sigma * torch.randn(args_cli.popsize, npar), hi), lo)
            fits = torch.tensor([score(p, f"it{it}") for p in pop])
            order = fits.argsort(descending=True)
            elite = pop[order[: max(args_cli.popsize // 3, 2)]]
            mean = elite.mean(0)
            sigma = torch.max(0.5 * sigma + 0.5 * elite.std(0), floor)
            if fits.max() > best_fit:
                best_fit, best = float(fits.max()), pop[order[0]].clone()
            print(f"[tune] iter {it} best_fit={best_fit:.3f} best={[round(x, 3) for x in decode(best)]}", flush=True)
        return

    if args_cli.move_stats:
        kp, repel, kd = [float(x) for x in torch.exp(mean)]
        ev.u.cfg.curriculum.move_target_distance = args_cli.move_distance
        ev.u.cfg.curriculum.move_planar = args_cli.move_planar
        ev.u.cfg.curriculum.move_curve_amp = args_cli.move_curve_amp
        for v in args_cli.move_speeds:
            for ff in (0.0, 1.0):
                rows = ev.evaluate_stats(kp, repel, kd, v, ff, args_cli.episodes, f"v={v},ff={ff}")
                with open(args_cli.out, "a") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                print(f"[stats] v={v} ff={ff} episodes={len(rows)} success={sum(r['success'] for r in rows) / max(len(rows), 1):.3f}", flush=True)
        return

    if args_cli.move:
        kp, repel, kd = [float(x) for x in torch.exp(mean)]
        ev.u.cfg.curriculum.move_target_distance = args_cli.move_distance
        ev.u.cfg.curriculum.move_planar = args_cli.move_planar
        ev.u.cfg.curriculum.move_curve_amp = args_cli.move_curve_amp
        for v in args_cli.move_speeds:
            for ff in (0.0, 1.0):
                res = ev.evaluate(kp, repel, kd, move_speed=v, ff=ff)
                row = dict(cond=f"move v={v} ff={ff}", n_agents=ev.u.num_drones, **res)
                print("[move]", json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items()}), flush=True)
                with open(args_cli.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
        return

    if args_cli.diag:
        for kp, repel, kd in [(2, 2, 1), (2, 1, 1), (2, 4, 1), (4, 2, 1), (1, 2, 1), (2, 2, 3)]:
            print("[diag]", json.dumps(diagnose(ev, kp, repel, kd)), flush=True)
        return

    if args_cli.sweep:
        kp, repel, kd = [float(x) for x in torch.exp(mean)]
        conds = [("clean", {})]
        conds += [(f"pos_noise={v}", dict(pos_noise=v)) for v in (0.05, 0.1, 0.2, 0.3)]
        conds += [(f"vel_noise={v}", dict(vel_noise=v)) for v in (0.1, 0.3, 0.6)]
        conds += [(f"dropout={v}", dict(dropout=v)) for v in (0.3, 0.6, 0.9)]
        conds += [(f"delay={v}", dict(delay=v)) for v in (2, 5, 10)]
        conds += [(f"max_steps={v}", dict(max_steps=v)) for v in (700, 500, 350)]
        conds += [("combo", dict(pos_noise=0.1, vel_noise=0.3, dropout=0.3, delay=2))]
        for name, cond in conds:
            res = ev.evaluate(kp, repel, kd, **cond)
            row = dict(cond=name, n_agents=ev.u.num_drones, kp=kp, repel=repel, kd=kd, **res)
            print("[sweep]", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            with open(args_cli.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        return

    run(mean, "init")
    best_fit, best = -1e9, mean.clone()
    for it in range(args_cli.iterations):
        pop = (mean + sigma * torch.randn(args_cli.popsize, 3)).clamp(LOG_LO, LOG_HI)
        fits = torch.tensor([run(p, f"it{it}") for p in pop])
        order = fits.argsort(descending=True)
        elite = pop[order[: max(args_cli.popsize // 3, 2)]]
        mean = elite.mean(0)
        sigma = (0.5 * sigma + 0.5 * elite.std(0)).clamp(min=0.05)
        if fits.max() > best_fit:
            best_fit, best = float(fits.max()), pop[order[0]].clone()
        print(f"[tune] iter {it} best_fit={best_fit:.3f} best={[round(float(x), 3) for x in torch.exp(best)]} "
              f"mean={[round(float(x), 3) for x in torch.exp(mean)]}", flush=True)


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
