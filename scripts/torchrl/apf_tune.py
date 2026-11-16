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
parser.add_argument("--deadband", action="store_true", help="Also tune a deadband: no attraction while a drone is within this fraction of the containment radius from the (preview) centre.")
parser.add_argument("--dead", type=float, default=0.0, help="With --final_params: deadband fraction of the containment radius.")
parser.add_argument("--warm", type=float, nargs="+", default=None, help="Start the ES mean at kp repel kd ff rscale [tau_p tau_f].")
parser.add_argument("--max_vel", type=float, default=None, help="Overrides controller.max_lin_vel_cmd (m/s per axis).")
parser.add_argument("--pure_policy", action="store_true", help="With --policy_ckpt: the policy output is the velocity command (no APF).")
parser.add_argument("--yaw_hold", action="store_true", help="Pure policy has 3 outputs; yaw rate is commanded 0.")
parser.add_argument("--dagger", type=int, default=0, metavar="ROUNDS", help="DAgger imitation of the APF given by --final_params/--preview_params: rollouts per round with expert-policy mixing, then supervised fit; writes --il_out checkpoints.")
parser.add_argument("--il_steps", type=int, default=1500, help="dagger: env steps per round.")
parser.add_argument("--il_fit", type=int, default=400, help="dagger: minibatch updates per round.")
parser.add_argument("--il_cap", type=int, default=2_000_000, help="dagger: aggregated dataset capacity (samples).")
parser.add_argument("--critic_steps", type=int, default=0, help="dagger: after the rounds, roll out the final policy with exploration noise for this many steps and fit a centralised critic on Monte-Carlo returns; saved with the policy and reward_rms in <il_out>_final.pt.")
parser.add_argument("--critic_noise", type=float, default=-2.0, help="dagger critic phase: policy log-std used when sampling actions.")
parser.add_argument("--critic_fit", type=int, default=3000, help="dagger critic phase: minibatch updates.")
parser.add_argument("--space_hinge", type=float, nargs=2, default=None, metavar=("DSEP", "POW"), help="Moving-task reward: spacing penalty starts at DSEP (m) with exponent POW.")
parser.add_argument("--il_out", type=str, default="il_ckpt")
parser.add_argument("--gain_dims", type=int, default=5, choices=[2, 5])
parser.add_argument("--gain_mod", type=float, default=0.0, help="Evaluate a gain-modulation policy (5 log-multipliers in [-1,1]*this on the final APF gains).")
parser.add_argument("--wp_horizon", type=int, default=0, help="Evaluate a waypoint-plan policy (K body-frame increments per drone).")
parser.add_argument("--wp_dt", type=float, default=0.2)
parser.add_argument("--wp_lookahead", type=float, default=0.3)
parser.add_argument("--wp_goal_bias", action="store_true")
parser.add_argument("--residual_kp", type=float, default=2.0, help="Waypoint mode: goal-bias attraction gain.")
parser.add_argument("--move_stats", action="store_true", help="Per-episode inter-agent distance statistics on the moving-sphere task.")
parser.add_argument("--sweep", action="store_true", help="Stress-test the --init gains over sensing/time-limit conditions.")
parser.add_argument("--gm_noise", action="store_true", help="Sample the policy action with its learned std (training-time exploration noise) instead of using the mean.")
parser.add_argument("--record", type=str, default=None, help="Record the first evaluated episode (env 0) to this video directory; use --num_envs 1 --episodes 1.")
parser.add_argument("--record_name", type=str, default="rec")
parser.add_argument("--record_steps", type=int, default=4000)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = args_cli.enable_cameras or args_cli.record is not None

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
    expand_gain_dims,
    compute_goal_bias,
    compute_swarm_gravity_baseline_action,
    compute_waypoint_plan_action,
)
from UavSwarm.tasks.direct.torchrl_swarm import paths as _pf  # noqa: E402
from UavSwarm.tasks.direct.torchrl_swarm.lidar import lidar_update, make_lidar  # noqa: E402

_EV = None
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
        if args_cli.record:
            cfg.sim.render.enable_translucency = True
            self.env = gym.make(args_cli.task, cfg=cfg, render_mode="rgb_array")
            self.env = gym.wrappers.RecordVideo(self.env, args_cli.record, step_trigger=lambda s: s == 0, video_length=args_cli.record_steps,
                                                name_prefix=args_cli.record_name, disable_logger=True)
        else:
            self.env = gym.make(args_cli.task, cfg=cfg)
        global _EV
        _EV = self
        self.u = self.env.unwrapped
        self._gm = None
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
            self.net = AttentionMAPPOPolicy(d - k * tdim, k, args_cli.gain_dims if args_cli.gain_mod > 0 else (3 if args_cli.yaw_hold else 4), token_dim=tdim, masked=tdim == 5)
        else:
            self.net = MAPPOPolicy(d, 3 if args_cli.yaw_hold else 4, (256, 256, 128))
        self.net.load_state_dict(sd)
        self.net.to(self.u.device).eval()

    def _policy_action(self):
        obs = torch.stack([self._obs[n] for n in self.names], dim=1).to(self.u.device)
        mean, std = self.net(obs)
        if args_cli.gm_noise:
            mean = mean + std * torch.randn_like(mean)
        return torch.tanh(mean)

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
                self._gm = args_cli.gain_mod * expand_gain_dims(self._policy_action(), args_cli.gain_dims)
                act = compute_apf_gain_mod_action(u, torch.exp(self._gm), base)
                self._obs, self._rew, _, _, _ = self.env.step({n: act[:, i] for i, n in enumerate(self.names)})
                return None
            if self.net is not None and args_cli.pure_policy:
                act = self._policy_action()
                if args_cli.yaw_hold:
                    act = torch.cat([act, torch.zeros_like(act[..., :1])], dim=-1)
                self._obs, self._rew, _, _, _ = self.env.step({n: act[:, i] for i, n in enumerate(self.names)})
                return None
            act = compute_swarm_gravity_baseline_action(self.u, kp, repel, ff_gain=ff)
            if self.net is not None:
                act = (act + args_cli.residual_scale * self._policy_action()).clamp(-1.0, 1.0)
            self._obs, self._rew, _, _, _ = self.env.step({n: act[:, i] for i, n in enumerate(self.names)})
            return None
        if getattr(self.u, "_lidar", None) is not None:
            if not self.u._lidar.get("auto"):
                lidar_update(self.u)
        elif self.u._apf_sense is not None:
            apf_sense_update(self.u)
        act = compute_swarm_gravity_baseline_action(self.u, kp, repel, ff_gain=ff)
        res = self.env.step({n: act[:, i] for i, n in enumerate(self.names)})
        self._rew = res[1]
        return res

    def _aim(self):
        u = self.u
        c = torch.stack([r.data.root_pos_w for r in u._robots], dim=1)[0].mean(0)
        d = u._target_vel_w[0, :2] if float(u._target_vel_w[0, :2].norm()) > 0.05 else u._target_dir_w[0, :2]
        d = d / d.norm().clamp(min=1e-6)
        eye = c.clone()
        eye[:2] += torch.stack([-d[1], d[0]]) * 5.0
        eye[2] += 2.5
        u.sim.set_camera_view(eye.tolist(), c.tolist())

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

    def _critic_phase(self, net, out_prefix):
        from mappo_torchl import CentralizedCritic, RunningMeanStd

        u, dev, a = self.u, self.u.device, args_cli
        E, D = u.num_envs, u.num_drones
        gamma = 0.99
        std_a = math.exp(a.critic_noise)
        with open(a.config) as f:
            hs = yaml.safe_load(f)["models"]["critic"]["hidden_sizes"]
        critic = CentralizedCritic(u.cfg.state_space, hs).to(dev)
        rms, run_ret = RunningMeanStd(device=dev), torch.zeros(E, D, device=dev)
        S, R, Dn = [], [], []
        self._obs, _ = self.env.reset()
        u.episode_length_buf[:] = 0
        for _ in range(a.critic_steps):
            obs = torch.stack([self._obs[n] for n in self.names], dim=1).to(dev)
            S.append(u._get_states().detach().cpu())
            with torch.no_grad():
                act3 = torch.tanh(net(obs)[0] + std_a * torch.randn(E, D, 3, device=dev))
            act = torch.cat([act3, torch.zeros_like(act3[..., :1])], dim=-1)
            self._obs, rew, _, _, _ = self.env.step({n: act[:, i] for i, n in enumerate(self.names)})
            r = torch.stack([rew[n] for n in self.names], dim=1).to(dev)
            done = (u._last_terminated | u._last_timed_out).float().view(E, 1)
            run_ret = torch.nan_to_num(run_ret * gamma * (1.0 - done) + r)
            rms.update(run_ret.reshape(-1))
            R.append(r.mean(1).cpu())
            Dn.append(done.view(E).cpu())
        R, Dn, S = torch.stack(R), torch.stack(Dn), torch.stack(S)  # (T, E), (T, E), (T, E, sd)
        G = torch.zeros_like(R)
        nxt = torch.zeros(E)
        for t in range(R.shape[0] - 1, -1, -1):
            nxt = R[t] + gamma * (1.0 - Dn[t]) * nxt
            G[t] = nxt
        keep = R.shape[0] - 300
        std = float(torch.sqrt(rms.var + 1e-8))
        n_hold = max(1, E // 8)
        Xtr, Ytr = S[:keep, : E - n_hold].reshape(-1, S.shape[-1]), (G[:keep, : E - n_hold] / std).reshape(-1)
        Xte, Yte = S[:keep, E - n_hold:].reshape(-1, S.shape[-1]).to(dev), (G[:keep, E - n_hold:] / std).reshape(-1).to(dev)
        opt = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-4)
        ev = lambda X, Y: 1.0 - float((critic(X).squeeze(-1) - Y).var() / Y.var())
        best_ev, best_sd = -1e9, None
        for it in range(a.critic_fit):
            b = torch.randint(0, Xtr.shape[0], (4096,))
            loss = torch.nn.functional.smooth_l1_loss(critic(Xtr[b].to(dev)).squeeze(-1), Ytr[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (it + 1) % 250 == 0 or it + 1 == a.critic_fit:
                with torch.no_grad():
                    ev_te = ev(Xte, Yte)
                    sub = torch.randint(0, Xtr.shape[0], (20000,))
                    ev_tr = ev(Xtr[sub].to(dev), Ytr[sub].to(dev))
                if ev_te > best_ev:
                    best_ev, best_sd = ev_te, {k_: v_.detach().clone() for k_, v_ in critic.state_dict().items()}
                print(f"[critic] it {it + 1} loss {float(loss):.4f} ev_train {ev_tr:.3f} ev_heldout {ev_te:.3f} best {best_ev:.3f} "
                      f"target_std {float(Ytr.std()):.3f} ret_std {std:.3f}", flush=True)
        critic.load_state_dict(best_sd)
        pol = {"module.0.module." + kk: v.cpu() for kk, v in net.state_dict().items()}
        crit = {"module." + kk: v.cpu() for kk, v in critic.state_dict().items()}
        torch.save({"policy": pol, "critic": crit, "reward_rms": {"mean": float(rms.mean), "var": float(rms.var), "count": float(rms.count)}},
                   f"{out_prefix}_final.pt")

    def dagger(self, kp, repel, kd, ff, rounds, steps, fit_iters, out_prefix):
        from mappo_torchl import AttentionMAPPOPolicy

        if args_cli.space_hinge:
            from UavSwarm.tasks.direct.torchrl_swarm import moving_task as _mt

            _mt.RW["d_sep"], _mt.RW["space_pow"] = args_cli.space_hinge

        u, dev, a = self.u, self.u.device, args_cli
        E, D = u.num_envs, u.num_drones
        u.cfg.swarm_cfg.apf_damping_gain = kd
        cur = u.cfg.curriculum
        speed = a.move_speed_range[1] if a.move_speed_range else a.move_speeds[0]
        cur.move_target_speed = speed
        u.cfg.episode_length_s = cur.moving_episode_s() if not cur.move_paths else (cur.move_path_len / speed + cur.move_pause_budget_s + cur.move_settle_s + 0.5)
        k = u.cfg.moving_obs_k
        d = u.cfg.single_observation_space
        net = AttentionMAPPOPolicy(d - k * 5, k, 3, token_dim=5, masked=True).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        radius = cur.get_containment_radius(D, u.cfg.swarm_cfg.min_safe_distance)
        cap = args_cli.il_cap
        X = torch.zeros(cap, d, device=dev)
        Y = torch.zeros(cap, 3, device=dev)
        n_data = ptr = 0
        self._obs, _ = self.env.reset()
        u.episode_length_buf[:] = 0
        for r in range(rounds):
            beta = 0.5 ** r if r < 4 else 0.0
            packed = steps_done = collisions = 0
            err = 0.0
            for _ in range(steps):
                obs = torch.stack([self._obs[n] for n in self.names], dim=1).to(dev)  # (E, D, d)
                with torch.no_grad():
                    exp_a = compute_swarm_gravity_baseline_action(u, kp, repel, ff_gain=ff).clamp(-1.0, 1.0)
                    pol_a = torch.tanh(net(obs)[0])
                    err += float((pol_a - exp_a[..., :3]).pow(2).mean())
                use_exp = (torch.rand(E, 1, 1, device=dev) < beta).float()
                act3 = use_exp * exp_a[..., :3] + (1 - use_exp) * pol_a
                m = E * D
                idx = (ptr + torch.arange(m, device=dev)) % cap
                X[idx], Y[idx] = obs.reshape(m, d), exp_a[..., :3].reshape(m, 3)
                ptr, n_data = (ptr + m) % cap, min(cap, n_data + m)
                act = torch.cat([act3, torch.zeros_like(act3[..., :1])], dim=-1)
                self._obs, _, _, _, _ = self.env.step({n: act[:, i] for i, n in enumerate(self.names)})
                pos = torch.stack([rb.data.root_pos_w for rb in u._robots], dim=0)
                dist = (u._shared_target_w.unsqueeze(0) - pos).norm(dim=2)
                pair = (pos.unsqueeze(1) - pos.unsqueeze(0)).norm(dim=-1) + 10 * torch.eye(D, device=dev).unsqueeze(-1)
                nn = pair.min(dim=1).values.min(dim=0).values
                packed += float(((dist <= radius).all(0) & (nn >= 0.5)).float().mean())
                collisions += int((u._last_terminated).sum())
                steps_done += 1
            loss_sum = 0.0
            for _ in range(fit_iters):
                b = torch.randint(0, n_data, (4096,), device=dev)
                loss = (torch.tanh(net(X[b])[0]) - Y[b]).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += float(loss)
            print(f"[dagger] round {r} beta {beta:.2f} packed {packed / steps_done:.3f} terminated {collisions} act_mse {err / steps_done:.4f} "
                  f"fit_loss {loss_sum / fit_iters:.4f} data {n_data}", flush=True)
            torch.save({"policy": {"module.0.module." + kk: v.cpu() for kk, v in net.state_dict().items()}}, f"{out_prefix}_r{r}.pt")
        if args_cli.critic_steps > 0:
            self._critic_phase(net, out_prefix)

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
                 "jit_mv", "jit_st", "spd_mv", "spd_st", "cerr_mv", "cerr_st", "clag_mv", "clag_max", "maxd_mv", "rew_sum", "n_rew", "rt_contain", "rt_centre", "rt_space", "rt_smooth", "rt_packed", "rt_fail"]
        acc = {}

        def reset_acc(mask=None):
            for k in names + ["min_nn"]:
                fill = 99.0 if k == "min_nn" else 0.0
                if mask is None:
                    acc[k] = torch.full((E,), fill, device=dev)
                else:
                    acc[k][mask] = fill

        gm_sum, gm_sd = torch.zeros(E, 5, device=dev), torch.zeros(E, 5, device=dev)
        reset_acc()
        self._obs, _ = self.env.reset()
        self._wp_prev = None
        u.episode_length_buf[:] = 0
        rows, step = [], 0
        while len(rows) < episodes and step < max(12000, int(6 * u.cfg.episode_length_s / self.step_dt)):
            fam_prev, prof_prev, spd_prev = u._path_fam.clone(), u._path_prof.clone(), u._path_speed.clone()
            self._step(kp, repel, ff)
            step += 1
            if a.record and step % 10 == 1:
                self._aim()
            done = u._last_terminated | u._last_timed_out
            alive = ~done
            if self._gm is not None:
                gm_sum += self._gm.mean(1) * alive.unsqueeze(-1)
                gm_sd += self._gm.std(1) * alive.unsqueeze(-1)
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
            if getattr(self, "_rew", None) is not None:
                acc["rew_sum"] += torch.stack([self._rew[n] for n in self.names]).mean(0).to(dev)
                acc["n_rew"] += 1.0
                for k_, v_ in getattr(u, "_rew_terms", {}).items():
                    acc["rt_" + k_] += v_
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
                        reward=g("rew_sum") / max(g("n_rew"), 1.0),
                        rew_terms={k_: g("rt_" + k_) / max(g("n_rew"), 1.0) for k_ in ("contain", "centre", "space", "smooth", "packed", "fail")},
                        success=bool(packed_all >= 0.9 and not coll),
                        **({"gm": (gm_sum[e] / n_all).tolist(), "gm_sd": (gm_sd[e] / n_all).tolist()} if self._gm is not None else {}),
                    ))
                reset_acc(done)
                gm_sum[done] = 0.0
                gm_sd[done] = 0.0
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

        def run_set(kp, repel, kd, ff, rscale, speeds, episodes, tag, preview=None, dead=0.0):
            ev.u._apf_radius_scale = rscale
            ev.u._apf_preview = preview
            ev.u._apf_dead = dead
            out = []
            for v in speeds:
                out += ev.evaluate_stats(kp, repel, kd, v, ff, episodes, f"{tag},v={v}")
            for r in out:
                r.update(kp=kp, repel=repel, kd=kd, ff_gain=ff, rscale=rscale, preview=preview, dead=dead)
            return out

        if args_cli.dagger:
            kp, repel, kd, ff, rscale = args_cli.final_params
            ev.u._apf_radius_scale, ev.u._apf_dead = rscale, 0.0
            ev.u._apf_preview = tuple(args_cli.preview_params) if args_cli.preview_params else None
            ev.dagger(kp, repel, kd, ff, args_cli.dagger, args_cli.il_steps, args_cli.il_fit, args_cli.il_out)
            return

        if args_cli.final_params:
            kp, repel, kd, ff, rscale = args_cli.final_params
            rows = run_set(kp, repel, kd, ff, rscale, args_cli.move_speeds, args_cli.episodes, "tuned",
                           tuple(args_cli.preview_params) if args_cli.preview_params else None, args_cli.dead)
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
        if args_cli.deadband:
            lo, hi = torch.cat([lo, torch.tensor([0.0])]), torch.cat([hi, torch.tensor([0.9])])
            mean, sigma = torch.cat([mean, torch.tensor([0.3])]), torch.cat([sigma, torch.tensor([0.2])])
            floor = torch.cat([floor, torch.tensor([0.03])])
        if args_cli.warm:
            w = args_cli.warm
            mean[:5] = torch.tensor([math.log(w[0]), math.log(w[1]), math.log(w[2]), w[3], math.log(w[4])])
            if args_cli.preview and len(w) >= 7:
                mean[5:7] = torch.tensor(w[5:7])
        npar = len(mean)
        n_pv = 2 if args_cli.preview else 0

        def decode(p):
            base = (float(p[0].exp()), float(p[1].exp()), float(p[2].exp()), float(p[3]), float(p[4].exp()))
            return base + tuple(float(x) for x in p[5 : 5 + n_pv]), (float(p[5 + n_pv]) if args_cli.deadband else 0.0)

        def score(p, tag):
            (kp, repel, kd, ff, rscale, *pv), dead = decode(p)
            rows = run_set(kp, repel, kd, ff, rscale, args_cli.tune_speeds, args_cli.episodes, tag, tuple(pv) if pv else None, dead)
            per_v = {}
            for r in rows:
                per_v.setdefault(r["v"], []).append(r)
            fit = sum(sum(x["packed_all"] - 2.0 * x["coll"] for x in g) / len(g) for g in per_v.values()) / len(per_v)
            summ = dict(
                tag=tag, kp=kp, repel=repel, kd=kd, ff=ff, rscale=rscale, tp=(pv[0] if pv else None), tf=(pv[1] if pv else None), dead=dead, fit=fit,
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
            print(f"[tune] iter {it} best_fit={best_fit:.3f} best={decode(best)}", flush=True)
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

    if args_cli.record and _EV is not None:
        _EV.env.close()
    os._exit(0)
