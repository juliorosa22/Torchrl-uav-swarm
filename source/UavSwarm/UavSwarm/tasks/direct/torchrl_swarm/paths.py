"""Randomised sphere-centre trajectories for the moving-packing task: path families, speed profiles, curvature and altitude limits."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

TRAIN_FAMILIES = ("line", "curve", "slalom", "arc", "hairpin", "spline")
HELD_OUT_FAMILIES = ("figure8", "coaster", "helix")
ALL_FAMILIES = TRAIN_FAMILIES + HELD_OUT_FAMILIES
PROFILES = ("const", "ramp", "stopgo")
DS = 0.025  # uniform arc-length spacing (m)


def _sizes(L: float) -> tuple[int, int]:
    """(uniform samples M, dense parametric samples M0) enough for a path of length L with scale-up headroom."""
    return max(1600, int(2.0 * L / DS)), max(1200, int(60 * L))


def _u(n, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(n, device=dev)


def _sgn(n, dev):
    return torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)


def _polyline(corners: torch.Tensor, m0: int) -> torch.Tensor:
    seg = (corners[:, 1:] - corners[:, :-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(dim=1)], dim=1)
    q = torch.linspace(0.0, 1.0, m0, device=corners.device).unsqueeze(0) * cum[:, -1:]
    idx = torch.searchsorted(cum.contiguous(), q.contiguous(), right=True).clamp(1, corners.shape[1] - 1)
    c0, c1 = cum.gather(1, idx - 1), cum.gather(1, idx)
    w = ((q - c0) / (c1 - c0).clamp(min=1e-9)).unsqueeze(-1)
    p0 = corners.gather(1, (idx - 1).unsqueeze(-1).expand(-1, -1, 3))
    p1 = corners.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
    return p0 + w * (p1 - p0)


def _walk(lengths, yaw_inc, pitch):
    yaw = yaw_inc.cumsum(dim=1)
    d = torch.stack([pitch.cos() * yaw.cos(), pitch.cos() * yaw.sin(), pitch.sin()], dim=-1) * lengths.unsqueeze(-1)
    return torch.cat([torch.zeros_like(d[:, :1]), d.cumsum(dim=1)], dim=1)


def _gen(family: str, n: int, dev, L: float, rho_min: float, m0: int):
    """Dense local-frame points (n, m0, 3) and a smoothing length (m) for one family. Feature counts grow with L."""
    t = torch.linspace(0.0, 1.0, m0, device=dev).unsqueeze(0).expand(n, -1)
    Ld = 1.15 * L
    sigma = 0.0
    z0 = torch.zeros(n, 1, device=dev)
    if family == "line":
        pitch = torch.deg2rad(_u(n, -20.0, 20.0, dev)).unsqueeze(1)
        s = t * Ld
        pts = torch.stack([s * pitch.cos(), z0 + 0 * s, s * pitch.sin()], dim=-1)
    elif family == "curve":
        x = t * Ld
        lam = _u(n, 8.0, 14.0, dev).unsqueeze(1)
        y = _sgn(n, dev).unsqueeze(1) * _u(n, 0.6, 1.5, dev).unsqueeze(1) * torch.sin(2 * math.pi * x / lam)
        pts = torch.stack([x, y, 0 * x], dim=-1)
    elif family == "slalom":
        x = t * Ld
        lam = _u(n, 4.0, 7.0, dev).unsqueeze(1)
        y = _sgn(n, dev).unsqueeze(1) * _u(n, 0.5, 1.2, dev).unsqueeze(1) * torch.sin(2 * math.pi * x / lam)
        pts = torch.stack([x, y, 0 * x], dim=-1)
    elif family == "arc":
        r = _u(n, 1.2, 3.5, dev).unsqueeze(1)
        phi = t * (Ld / r)
        climb = _sgn(n, dev).unsqueeze(1) * _u(n, 0.0, 1.5, dev).unsqueeze(1) * max(1.0, L / 10.0)
        pts = torch.stack([r * phi.sin(), _sgn(n, dev).unsqueeze(1) * r * (1 - phi.cos()), climb * phi / (2 * math.pi)], dim=-1)
    elif family == "hairpin":
        segs = max(4, round(L / 3.0))
        mask = (torch.rand(n, segs - 1, device=dev) < 0.6).float()
        mask[:, 0] = 1.0
        turn = _sgn(n * (segs - 1), dev).reshape(n, segs - 1) * torch.deg2rad(_u(n * (segs - 1), 60.0, 150.0, dev)).reshape(n, segs - 1) * mask
        yaw_inc = torch.cat([torch.zeros(n, 1, device=dev), turn], dim=1)
        lengths = (Ld / segs) * _u(n * segs, 0.8, 1.2, dev).reshape(n, segs)
        pitch = torch.deg2rad(_u(n * segs, -10.0, 10.0, dev)).reshape(n, segs)
        pts = _polyline(_walk(lengths, yaw_inc, pitch), m0)
        sigma = max(0.5, rho_min) * float(_u(1, 1.0, 1.5, dev))
    elif family == "spline":
        segs = max(5, round(L / 2.5))
        yaw_inc = torch.cat([torch.zeros(n, 1, device=dev), torch.deg2rad((40.0 * torch.randn(n, segs - 1, device=dev)).clamp(-80, 80))], dim=1)
        pitch = torch.deg2rad(_u(n * segs, -15.0, 15.0, dev)).reshape(n, segs)
        pts = _polyline(_walk((Ld / segs) * torch.ones(n, segs, device=dev), yaw_inc, pitch), m0)
        sigma = max(0.8, rho_min) * float(_u(1, 1.0, 1.3, dev))
    elif family == "figure8":
        a = _u(n, 1.0, 1.6, dev).unsqueeze(1)
        th = t * 2 * math.pi * max(2.0, Ld / (6.0 * 1.3))
        h = _u(n, 0.3, 0.7, dev).unsqueeze(1) * (torch.rand(n, 1, device=dev) < 0.5).float()
        pts = torch.stack([a * th.sin(), a * th.sin() * th.cos(), h * torch.sin(2 * th)], dim=-1)
    elif family == "coaster":
        x = t * Ld
        lam = _u(n, 4.0, 8.0, dev).unsqueeze(1)
        pts = torch.stack([x, 0 * x, _u(n, 0.4, 0.9, dev).unsqueeze(1) * torch.sin(2 * math.pi * x / lam)], dim=-1)
    elif family == "helix":
        rho = _u(n, 1.5, 2.5, dev).unsqueeze(1)
        turns = Ld / (2 * math.pi * rho) * _u(n, 0.8, 1.1, dev).unsqueeze(1)
        phi = 2 * math.pi * turns * t
        climb = _sgn(n, dev).unsqueeze(1) * _u(n, 1.0, 2.0, dev).unsqueeze(1) * turns
        pts = torch.stack([rho * (phi.cos() - 1), _sgn(n, dev).unsqueeze(1) * rho * phi.sin(), climb * t], dim=-1)
    else:
        raise ValueError(f"unknown path family {family!r}")
    return pts, sigma


def _resample(pts: torch.Tensor, m: int):
    seg = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(dim=1)], dim=1)
    q = (torch.arange(m, device=pts.device).float() * DS).unsqueeze(0).expand(pts.shape[0], -1)
    q = torch.minimum(q, cum[:, -1:]).contiguous()
    idx = torch.searchsorted(cum.contiguous(), q, right=True).clamp(1, pts.shape[1] - 1)
    c0, c1 = cum.gather(1, idx - 1), cum.gather(1, idx)
    w = ((q - c0) / (c1 - c0).clamp(min=1e-9)).unsqueeze(-1)
    ex = lambda i: i.unsqueeze(-1).expand(-1, -1, 3)
    pa, pb = pts.gather(1, ex(idx - 1)), pts.gather(1, ex(idx))
    return pa + w * (pb - pa), cum[:, -1]


def _smooth(pts: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return pts
    r = int(3 * sigma / DS)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=pts.device).float() * DS / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    x = F.pad(pts.permute(0, 2, 1).reshape(-1, 1, pts.shape[1]), (r, r), mode="replicate")
    return F.conv1d(x, k).reshape(pts.shape[0], 3, -1).permute(0, 2, 1)


def _max_curvature(pts: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    h = 3
    a = (pts[:, 2 * h:] - pts[:, : -2 * h]) / (2 * h * DS)
    b = (pts[:, 2 * h:] - 2 * pts[:, h:-h] + pts[:, : -2 * h]) / (h * DS) ** 2
    kappa = torch.cross(a, b, dim=-1).norm(dim=-1) / a.norm(dim=-1).pow(3).clamp(min=1e-3)
    return (kappa * valid[:, h:-h]).max(dim=1).values


def _finalise(pts, sigma, L, rho_min, z_lo, z_hi, m):
    n, dev = pts.shape[0], pts.device
    pts, total = _resample(pts, m)
    pts = _smooth(pts, sigma)
    pts, total = _resample(pts, m)
    idx = torch.arange(m, device=dev).float().unsqueeze(0) * DS
    kmax = _max_curvature(pts, (idx <= torch.minimum(total, torch.full_like(total, L)).unsqueeze(1)).float())
    f = (rho_min * kmax).clamp(1.0, 6.0).view(n, 1, 1)
    pts = pts[:, :1] + f * (pts - pts[:, :1])
    pts, total = _resample(pts, m)
    length = torch.minimum(total, torch.full_like(total, L))
    yaw = _u(n, 0.0, 2 * math.pi, dev)
    c, s = yaw.cos().unsqueeze(1), yaw.sin().unsqueeze(1)
    pts = torch.stack([c * pts[..., 0] - s * pts[..., 1], s * pts[..., 0] + c * pts[..., 1], pts[..., 2]], dim=-1)
    valid = (idx <= length.unsqueeze(1)).float().unsqueeze(-1)
    cnt = valid.sum(dim=1).clamp(min=1)
    mid = (pts * valid).sum(dim=1) / cnt
    pts = pts - torch.cat([mid[:, :2], torch.zeros(n, 1, device=dev)], dim=1).unsqueeze(1)
    zmin = torch.where(valid.squeeze(-1) > 0, pts[..., 2], torch.full_like(pts[..., 2], 1e9)).min(dim=1).values
    zmax = torch.where(valid.squeeze(-1) > 0, pts[..., 2], torch.full_like(pts[..., 2], -1e9)).max(dim=1).values
    allowed = z_hi - z_lo
    span = zmax - zmin
    comp = torch.where(span > allowed, allowed / span.clamp(min=1e-6), torch.ones_like(span))
    zc = (zmin + zmax) / 2
    z = zc.unsqueeze(1) + comp.unsqueeze(1) * (pts[..., 2] - zc.unsqueeze(1))
    zmin2 = zc + comp * (zmin - zc)
    z = z - zmin2.unsqueeze(1) + (z_lo + (allowed - span * comp).clamp(min=0).unsqueeze(1) * torch.rand(n, 1, device=dev))
    return torch.cat([pts[..., :2], z.unsqueeze(-1)], dim=-1), length


def speed_multiplier(prof: torch.Tensor, n_steps: int, dt: float) -> torch.Tensor:
    """(n, n_steps) multiplier on the nominal speed for each profile id (const, ramp, stopgo)."""
    n, dev = prof.shape[0], prof.device
    t = (torch.arange(n_steps, device=dev).float() * dt).unsqueeze(0)
    m = torch.ones(n, n_steps, device=dev)
    ramp = (0.25 + t / 3.0).clamp(max=1.0)
    t1 = _u(n, 2.5, 6.0, dev).unsqueeze(1)
    w1 = _u(n, 1.0, 2.0, dev).unsqueeze(1)
    t2 = t1 + w1 + _u(n, 1.5, 3.0, dev).unsqueeze(1)
    w2 = _u(n, 1.0, 2.0, dev).unsqueeze(1) * (torch.rand(n, 1, device=dev) < 0.5).float()

    def box(t0, w):
        return torch.sigmoid((t - t0) / 0.15) - torch.sigmoid((t - t0 - w) / 0.15)

    stopgo = (1.0 - box(t1, w1) - box(t2, w2)).clamp(min=0.0)
    m = torch.where((prof == 1).unsqueeze(1), ramp.expand(n, -1), m)
    return torch.where((prof == 2).unsqueeze(1), stopgo, m)


def generate(fam_ids: torch.Tensor, prof_ids: torch.Tensor, speed: float, dt: float, n_steps: int, L: float,
             a_max: float, z_lo: float, z_hi: float):
    """Sphere-centre trajectories (n, n_steps, 3) at control rate, moving-step count (n,), path length (n,)."""
    n, dev = fam_ids.shape[0], fam_ids.device
    rho_min = max(0.8, speed**2 / a_max)
    M, m0 = _sizes(L)
    pts = torch.zeros(n, M, 3, device=dev)
    length = torch.zeros(n, device=dev)
    for f, name in enumerate(ALL_FAMILIES):
        sel = (fam_ids == f).nonzero().flatten()
        if sel.numel() == 0:
            continue
        dense, sigma = _gen(name, sel.numel(), dev, L, rho_min, m0)
        pts[sel], length[sel] = _finalise(dense, sigma, L, rho_min, z_lo, z_hi, M)
    s = torch.cumsum(speed * speed_multiplier(prof_ids, n_steps, dt) * dt, dim=1)
    s = torch.minimum(s - speed * dt, length.unsqueeze(1)).clamp(min=0.0)
    pos = (s / DS).clamp(max=M - 1.001)
    i0 = pos.floor().long()
    w = (pos - i0).unsqueeze(-1)
    ex = lambda i: i.unsqueeze(-1).expand(-1, -1, 3)
    traj = pts.gather(1, ex(i0)) * (1 - w) + pts.gather(1, ex(i0 + 1)) * w
    moving = (s < length.unsqueeze(1) - 1e-6).sum(dim=1)
    k = torch.exp(-0.5 * (torch.arange(-6, 7, device=dev).float() / 2.0) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    x = F.pad(traj.permute(0, 2, 1).reshape(-1, 1, n_steps), (6, 6), mode="replicate")
    traj = F.conv1d(x, k).reshape(n, 3, n_steps).permute(0, 2, 1)
    return traj, moving, length
