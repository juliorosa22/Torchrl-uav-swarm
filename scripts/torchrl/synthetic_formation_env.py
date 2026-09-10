"""Synthetic V-formation variant of SyntheticSwarmEnv -- isolates whether the *shared,
Hungarian-assigned formation slot* structure (not UAV flight dynamics) is what blocks
convergence on the real Formation-TorchRL-UAVSwarm task.

Identical to SyntheticSwarmEnv in every other respect (point-mass kinematics, individual
per-agent reward, centralized critic seeing concatenated obs) except how goals are
generated at reset: instead of independent random per-agent goals, every agent's goal is
a slot in one shared V-formation (random center + heading each episode), assigned via
linear_sum_assignment (Hungarian) to minimize total swarm travel distance -- the same
mechanism source/UavSwarm/.../torchrl_swarm/curriculum.py::set_formation_positions uses
for the real task.

One variable changed at a time (see [[diagnostic-methodology]]): if this still converges
cleanly, formation/slot-coordination structure is ruled out as the real task's blocker and
the difficulty is UAV-flight-dynamics-specific; if it does NOT converge, the
coordination/assignment structure itself is a real, physics-independent difficulty.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from synthetic_swarm_env import SyntheticSwarmEnv


class SyntheticFormationEnv(SyntheticSwarmEnv):
    def __init__(self, *args, formation_spacing: float = 1.0, **kwargs):
        self.formation_spacing = formation_spacing
        super().__init__(*args, **kwargs)

    def _formation_offsets(self) -> torch.Tensor:
        """Fixed inverted-V slot offsets (n_agents, 2) relative to the leader (slot 0 at
        the origin), alternating left/right behind it -- same qualitative shape as the
        real task's V-formation. Scaled by formation_spacing.
        """
        offsets = torch.zeros(self.num_agents, 2)
        for i in range(1, self.num_agents):
            k = (i + 1) // 2
            side = 1.0 if i % 2 == 1 else -1.0
            offsets[i, 0] = -k * self.formation_spacing        # behind the leader
            offsets[i, 1] = side * k * self.formation_spacing  # alternating left/right
        return offsets

    def _sample_spawn_and_goal(self, mask: torch.Tensor):
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        n = idx.numel()

        # Scatter spawn -- unchanged from the base env.
        self.pos[idx] = (torch.rand(n, self.num_agents, 2, device=self.device) * 2 - 1) * self.bound
        self.last_vel[idx] = 0.0
        self.t[idx] = 0

        # Shared V-formation target: random center + heading per env, fixed relative shape.
        center = (torch.rand(n, 2, device=self.device) * 2 - 1) * self.bound * 0.5
        heading = torch.rand(n, device=self.device) * 2 * np.pi
        offsets = self._formation_offsets().to(self.device)  # (n_agents, 2)

        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        rot = torch.stack([
            torch.stack([cos_h, -sin_h], dim=-1),
            torch.stack([sin_h, cos_h], dim=-1),
        ], dim=1)  # (n, 2, 2)

        rotated = torch.einsum("nij,aj->nai", rot, offsets)  # (n, n_agents, 2)
        slots = rotated + center.unsqueeze(1)  # (n, n_agents, 2)

        # Hungarian assignment per env: minimize total swarm travel distance from spawn
        # to slot (scipy has no batched solver, but this only runs at reset, not per step).
        spawn_np = self.pos[idx].cpu().numpy()
        slots_np = slots.cpu().numpy()
        assigned = np.zeros_like(slots_np)
        for b in range(n):
            cost = np.linalg.norm(spawn_np[b][:, None, :] - slots_np[b][None, :, :], axis=-1)
            row, col = linear_sum_assignment(cost)
            assigned[b, row] = slots_np[b, col]

        self.goal[idx] = torch.as_tensor(assigned, device=self.device, dtype=torch.float32)
