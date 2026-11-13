"""Moving-sphere task: decentralised observation, reward and cfg wiring for the learned policy."""

from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply_inverse

from .curriculum import moving_preview
from .lidar import lidar_update

PREVIEW_TAUS = (0.5, 1.0, 1.5)
N_FIXED = 9 + 8 + 3 * len(PREVIEW_TAUS) + 4
RW = dict(contain=1.0, centre=0.1, space=1.0, d_sep=0.55, sep_eval=0.5, packed=0.5, smooth=0.05, fail=10.0)


def token_dim(sense: str) -> int:
    return 6 if sense == "oracle" else 5


def obs_dim(k: int, sense: str) -> int:
    return N_FIXED + k * token_dim(sense)


def configure_cfg(cfg, sense: str = "oracle", k: int = 4, num_agents: int | None = None, lidar_cfg: dict | None = None) -> None:
    n = num_agents or cfg.num_agents
    if n != cfg.num_agents:
        cfg.num_agents = n
        cfg.possible_agents = [f"robot_{i}" for i in range(n)]
        cfg.action_spaces = {f"robot_{i}": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for i in range(n)}
    k = min(k, n - 1) if sense == "oracle" else k
    d = obs_dim(k, sense)
    cfg.moving_obs, cfg.moving_obs_sense, cfg.moving_obs_k, cfg.lidar_cfg = True, sense, k, lidar_cfg
    cfg.single_observation_space = d
    cfg.observation_spaces = {
        f"robot_{i}": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(d,)) for i in range(n)
    }
    cfg.state_space = n * d + 6 * n


def _stack(env, attr):
    return torch.stack([getattr(rob.data, attr) for rob in env._robots], dim=0)


def _to_body(quats, vec):
    d, e = vec.shape[:2]
    return quat_apply_inverse(quats.reshape(-1, 4), vec.reshape(-1, 3)).reshape(vec.shape)


def _oracle_tokens(env, pos, vel, quats, k):
    rel = pos.permute(1, 0, 2).unsqueeze(1) - pos.permute(1, 0, 2).unsqueeze(2)  # (E, i, j, 3): pos_j - pos_i
    rvel = vel.permute(1, 0, 2).unsqueeze(1) - vel.permute(1, 0, 2).unsqueeze(2)
    dist = rel.norm(dim=-1) + 1e3 * torch.eye(env.num_drones, device=pos.device).unsqueeze(0)
    idx = dist.topk(k, dim=2, largest=False).indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    near = torch.cat([rel.gather(2, idx), rvel.gather(2, idx)], dim=-1).permute(1, 0, 2, 3)  # (D, E, k, 6)
    q = quats.unsqueeze(2).expand(-1, -1, k, -1)
    return torch.cat([_to_body(q, near[..., :3]), _to_body(q, near[..., 3:])], dim=-1)


def _lidar_tokens(env, quats, k):
    lid = env._lidar
    rng, dirs = lid["scan"]
    idx = rng.topk(k, dim=2, largest=False).indices
    r = rng.gather(2, idx)
    d = dirs.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, 3))
    valid = r.isfinite()
    db = _to_body(quats.unsqueeze(2).expand(-1, -1, k, -1), d)
    tok = torch.cat([(r / lid["rng_max"]).nan_to_num(posinf=0.0).unsqueeze(-1), db, valid.float().unsqueeze(-1)], dim=-1)
    return tok * valid.float().unsqueeze(-1)


def build_obs(env) -> torch.Tensor:
    """(D, E, obs_dim) per-drone observation: own state, sphere, path preview, previous action, neighbour tokens."""
    cfg = env.cfg
    pos, vel_w, quats = _stack(env, "root_pos_w"), _stack(env, "root_lin_vel_w"), _stack(env, "root_quat_w")
    lin_b, ang_b, grav = _stack(env, "root_lin_vel_b"), _stack(env, "root_ang_vel_b"), _stack(env, "projected_gravity_b")
    centre = env._shared_target_w.unsqueeze(0)
    radius = cfg.curriculum.get_containment_radius(env.num_drones, cfg.swarm_cfg.min_safe_distance)
    D, E = pos.shape[:2]
    off_b = _to_body(quats, centre - pos) / radius
    sv_b = _to_body(quats, env._target_vel_w.unsqueeze(0).expand(D, -1, -1))
    hold = (env._target_left <= 0).float().unsqueeze(0).unsqueeze(-1).expand(D, -1, 1)
    rad = torch.full((D, E, 1), radius, device=pos.device)
    prev = moving_preview(env, PREVIEW_TAUS)  # (E, T, 3)
    prev_b = torch.cat([_to_body(quats, (prev[:, t] - env._shared_target_w).unsqueeze(0).expand(D, -1, -1)) for t in range(prev.shape[1])], dim=-1)
    act = env._actions.permute(1, 0, 2)
    k = cfg.moving_obs_k
    if cfg.moving_obs_sense == "lidar":
        lidar_update(env)
        tok = _lidar_tokens(env, quats, k)
    else:
        tok = _oracle_tokens(env, pos, vel_w, quats, k)
    return torch.cat([lin_b, ang_b, grav, off_b, sv_b, rad, hold, prev_b, act, tok.reshape(D, E, -1)], dim=-1)


def build_state(env, obs: torch.Tensor) -> torch.Tensor:
    pos, vel = _stack(env, "root_pos_w"), _stack(env, "root_lin_vel_w")
    c = env._shared_target_w.unsqueeze(0)
    priv = torch.cat([pos - c, vel - env._target_vel_w.unsqueeze(0)], dim=-1).permute(1, 0, 2).reshape(env.num_envs, -1)
    return torch.cat([obs.permute(1, 0, 2).reshape(env.num_envs, -1), priv], dim=-1)


def get_rewards(env) -> dict[str, torch.Tensor]:
    cfg = env.cfg
    pos = _stack(env, "root_pos_w")
    D = env.num_drones
    radius = cfg.curriculum.get_containment_radius(D, cfg.swarm_cfg.min_safe_distance)
    dist = (pos - env._shared_target_w.unsqueeze(0)).norm(dim=-1)
    pair = (pos.unsqueeze(1) - pos.unsqueeze(0)).norm(dim=-1)
    eye = torch.eye(D, dtype=torch.bool, device=pos.device).unsqueeze(-1)
    viol = ((RW["d_sep"] - pair) / RW["d_sep"]).clamp(min=0.0).masked_fill(eye, 0.0)
    nn = pair.masked_fill(eye, 1e3).min(dim=1).values.min(dim=0).values
    inside_all = (dist <= radius).all(dim=0)
    packed = (inside_all & (nn >= RW["sep_eval"])).float()
    smooth = (env._actions - env._prev_actions).pow(2).sum(dim=-1).transpose(0, 1)
    reasons = env._termination_reasons
    failed = (reasons["collision"] | reasons["out_of_bounds"] | reasons["inter_agent_collision"]).float()
    r = (
        -RW["contain"] * (dist - radius).clamp(min=0.0) / radius
        - RW["centre"] * dist / radius
        - RW["space"] * viol.pow(2).sum(dim=1)
        - RW["smooth"] * smooth
        + RW["packed"] * packed.unsqueeze(0)
        - RW["fail"] * failed.unsqueeze(0)
    )
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=-RW["fail"])
    dt = env.step_dt
    env._metrics.update(
        distance_to_goal=dist.mean(dim=0), mean_reward=r.mean(dim=0), min_neighbor_distance=nn,
        pack_progress=packed * dt, pack_all_inside=inside_all.float() * dt, pack_all_spaced=(nn >= RW["sep_eval"]).float() * dt,
    )
    return {f"robot_{i}": r[i] for i in range(D)}
