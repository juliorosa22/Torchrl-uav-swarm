"""Onboard lidar model: neighbours are visible only as ray hits (no identity, no velocity, occlusion)."""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply


def make_lidar(n_az=16, elev_deg=(-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0), rng_max=3.0, rng_min=0.1, radius=0.1, sig_a=0.02, sig_b=0.01,
               miss=0.02, dither_deg=0.5, period=1, delay=0) -> dict:
    return dict(n_az=n_az, elev=tuple(elev_deg), rng_max=rng_max, rng_min=rng_min, radius=radius, sig_a=sig_a,
                sig_b=sig_b, miss=miss, dither=math.radians(dither_deg), period=period, delay=delay, buf=[], t=0)


def beam_dirs_body(lid, device) -> torch.Tensor:
    az = torch.arange(lid["n_az"], device=device) * (2 * math.pi / lid["n_az"])
    rows = [e for e in lid["elev"] if abs(e) < 89.9]
    el = torch.tensor([math.radians(e) for e in rows], device=device)
    el, az = torch.meshgrid(el, az, indexing="ij")
    ring = torch.stack([el.cos() * az.cos(), el.cos() * az.sin(), el.sin()], dim=-1).reshape(-1, 3)
    poles = [[0.0, 0.0, math.copysign(1.0, e)] for e in lid["elev"] if abs(e) >= 89.9]
    return torch.cat([ring, torch.tensor(poles, device=device).reshape(-1, 3)], dim=0)


def _world_dirs(quat, b):
    D, E = quat.shape[:2]
    q = quat.unsqueeze(2).expand(D, E, b.shape[0], 4).reshape(-1, 4)
    return quat_apply(q, b.unsqueeze(0).unsqueeze(0).expand(D, E, -1, 3).reshape(-1, 3)).reshape(D, E, -1, 3)


def raycast(pos, quat, lid, b):
    """Nearest-hit range per beam, shape (D, E, B), inf where nothing is hit. Occlusion by nearest hit."""
    D = pos.shape[0]
    dirs = _world_dirs(quat, b)
    if lid["dither"] > 0:
        dirs = dirs + lid["dither"] * torch.randn_like(dirs)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    oc = pos.unsqueeze(0) - pos.unsqueeze(1)  # (i, j, E, 3): origin i -> target j
    t_ca = torch.einsum("ijek,iebk->ijeb", oc, dirs)
    d2 = (oc * oc).sum(-1).unsqueeze(-1) - t_ca * t_ca
    r2 = lid["radius"] ** 2
    t_hit = t_ca - (r2 - d2).clamp(min=0).sqrt()
    ok = (t_ca > 0) & (d2 <= r2) & ~torch.eye(D, dtype=torch.bool, device=pos.device).view(D, D, 1, 1)
    return torch.where(ok, t_hit, torch.full_like(t_hit, float("inf"))).min(dim=1).values, dirs


def lidar_update(env) -> None:
    """Refresh env._lidar['scan'] = (ranges (D,E,B) with inf for no return, world directions (D,E,B,3))."""
    lid = env._lidar
    pos = torch.stack([r.data.root_pos_w for r in env._robots], dim=0)
    quat = torch.stack([r.data.root_quat_w for r in env._robots], dim=0)
    fresh = (env.episode_length_buf == 0).view(1, -1, 1)
    lid["buf"] = [(torch.where(fresh, pos, p0), torch.where(fresh, quat.clone(), q0)) for p0, q0 in lid["buf"]]
    lid["buf"].append((pos, quat))
    del lid["buf"][: -(lid["delay"] + 1)]
    t = lid["t"]
    lid["t"] = t + 1
    fresh_e = env.episode_length_buf == 0
    if "held" in lid and t % lid["period"] != 0 and not bool(fresh_e.any()):
        lid["scan"] = lid["held"]
        return
    b = lid.get("_b")
    if b is None:
        b = lid["_b"] = beam_dirs_body(lid, pos.device)
    p_d, q_d = lid["buf"][0]
    rng, dirs = raycast(p_d, q_d, lid, b)
    hit = rng.isfinite()
    rng = rng + (lid["sig_a"] + lid["sig_b"] * rng.nan_to_num(posinf=0.0)) * torch.randn_like(rng)
    keep = hit & (rng >= lid["rng_min"]) & (rng <= lid["rng_max"]) & (torch.rand_like(rng) >= lid["miss"])
    rng = torch.where(keep, rng, torch.full_like(rng, float("inf")))
    new = (rng, dirs)
    if "held" in lid and not (t % lid["period"] == 0):
        sel = fresh_e.view(1, -1, 1)
        new = (torch.where(sel, rng, lid["held"][0]), torch.where(sel.unsqueeze(-1), dirs, lid["held"][1]))
    lid["held"] = lid["scan"] = new
