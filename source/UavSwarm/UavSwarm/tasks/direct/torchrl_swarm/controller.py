"""Low-level controllers for UAV swarm environments.

Implements a geometric SE(3) controller, a simplified PD velocity controller,
and the original direct force/torque mapping as drop-in interchangeable options.

The policy outputs high-level velocity commands; the controller handles
attitude stabilization and thrust computation, decoupling flight physics from
navigation learning.

Action semantics when controller type is 'geometric' or 'pd_velocity':
    [vx_b_cmd, vy_b_cmd, vz_b_cmd, yaw_rate_cmd] ∈ [-1, 1]
    scaled to max_lin_vel_cmd (m/s) and max_yaw_rate_cmd (rad/s).

Action semantics when controller type is 'direct' (original behavior):
    [thrust, mx, my, mz] ∈ [-1, 1] — direct force/torque normalization.
"""

from __future__ import annotations

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse


@configclass
class ControllerCfg:
    """Configuration for the low-level controller layer."""

    type: str = "geometric"
    """Controller type: 'geometric' | 'pd_velocity' | 'direct'.

    - 'geometric': SE(3) geometric controller (Lee et al. 2010). Policy outputs
      body-frame velocity + yaw rate commands. Globally stable at any attitude.
    - 'pd_velocity': Simplified PD velocity tracking. Approximate near hover only.
    - 'direct': Original direct force/torque mapping. Backward-compatible no-op.
    """

    # Velocity command limits
    max_lin_vel_cmd: float = 1.5
    """Maximum linear velocity command per axis in body frame (m/s)."""

    max_yaw_rate_cmd: float = 1.0
    """Maximum yaw rate command (rad/s)."""

    # Geometric controller gains (outer + inner loop)
    Kv: float = 2.0
    """Velocity tracking proportional gain (outer loop, world frame)."""

    KR: float = 0.002
    """Attitude error proportional gain (inner loop).
    Scaled for Crazyflie physical range: moment_scale=0.01 Nm, max eR ≈ pi/2 → KR ≈ 0.002."""

    KOmega: float = 0.0005
    """Angular velocity error derivative gain (inner loop).
    Scaled for Crazyflie physical range: max omega ≈ 20 rad/s → KOmega*20 < 0.01 Nm."""

    # PD velocity controller gains (used only when type='pd_velocity')
    kp_vel_z: float = 2.0
    """Vertical velocity P-gain (maps vz error to thrust perturbation)."""

    kp_vel_xy: float = 1.5
    """Horizontal velocity P-gain (maps vxy error to roll/pitch moments)."""

    kp_yaw_rate: float = 0.5
    """Yaw rate P-gain (maps yaw rate error to yaw moment)."""


# ---------------------------------------------------------------------------
# Pure-PyTorch math helpers
# ---------------------------------------------------------------------------

def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert (w, x, y, z) quaternion to 3×3 rotation matrix.

    Maps body-frame vectors to world-frame: ``v_world = R @ v_body``.

    Args:
        q: Quaternion tensor of shape (..., 4) in (w, x, y, z) order.

    Returns:
        Rotation matrix of shape (..., 3, 3).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    mat = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
    return mat.reshape(*q.shape[:-1], 3, 3)


def _vee(S: torch.Tensor) -> torch.Tensor:
    """Extract axial vector from a skew-symmetric matrix.

    Args:
        S: Skew-symmetric tensor of shape (..., 3, 3).

    Returns:
        Axial vector of shape (..., 3).
    """
    return torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], dim=-1)


# ---------------------------------------------------------------------------
# Controller implementations
# ---------------------------------------------------------------------------

def compute_geometric_controller(
    env,
    actions_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Geometric SE(3) controller for quadrotor velocity tracking.

    Implements the velocity-tracking geometric controller (Lee et al. 2010).
    The policy outputs body-frame velocity commands; the controller cascades:
      1. Outer loop: velocity P-control → desired force vector
      2. Desired attitude from force direction + desired yaw
      3. Inner loop: attitude + angular velocity error → moments

    All computation is batched over ``num_envs * num_drones``; no Python loops.

    Args:
        env: ``BaseSwarmEnv`` instance with ``cfg.controller`` of type ``ControllerCfg``.
        actions_tensor: Clamped policy actions of shape ``(num_envs, num_drones, 4)``
            with channels ``[vx_b, vy_b, vz_b, yaw_rate]`` ∈ [-1, 1].

    Returns:
        thrust: Body-frame force tensor of shape ``(num_envs, num_drones, 1, 3)``.
            Only the z-component (body up) is non-zero.
        moment: Body-frame torque tensor of shape ``(num_envs, num_drones, 1, 3)``.
    """
    cfg = env.cfg.controller
    num_envs, num_drones = env.num_envs, env.num_drones
    device = env.device
    N = num_envs * num_drones
    dt = env.cfg.sim.dt * env.cfg.decimation

    # Gather and flatten robot state to (N, ...)
    all_quats = torch.stack(
        [rob.data.root_quat_w for rob in env._robots], dim=1
    ).reshape(N, 4)
    all_vel_b = torch.stack(
        [rob.data.root_lin_vel_b for rob in env._robots], dim=1
    ).reshape(N, 3)
    all_omega_b = torch.stack(
        [rob.data.root_ang_vel_b for rob in env._robots], dim=1
    ).reshape(N, 3)

    # Guard against NaN/Inf from physics divergence (e.g. after collision).
    # Replace invalid quaternions with identity; clamp velocities to sane range.
    quat_finite = all_quats.isfinite().all(dim=-1, keepdim=True)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand_as(all_quats)
    all_quats = torch.where(quat_finite, all_quats, identity_quat)
    all_vel_b = torch.nan_to_num(all_vel_b, nan=0.0).clamp(-20.0, 20.0)
    all_omega_b = torch.nan_to_num(all_omega_b, nan=0.0).clamp(-20.0, 20.0)

    # Scale actions to physical commands
    acts = actions_tensor.reshape(N, 4)
    v_des_b = acts[:, :3] * cfg.max_lin_vel_cmd     # desired body-frame velocity (N, 3)
    yaw_rate_des = acts[:, 3] * cfg.max_yaw_rate_cmd  # desired yaw rate (N,)

    # Rotation matrix: v_world = R @ v_body
    R = _quat_to_matrix(all_quats)  # (N, 3, 3)

    # Transform velocities to world frame
    v_des_w = torch.bmm(R, v_des_b.unsqueeze(-1)).squeeze(-1)   # (N, 3)
    v_curr_w = torch.bmm(R, all_vel_b.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    # Outer-loop: desired acceleration = velocity P-gain + gravity compensation
    g_comp = torch.zeros(N, 3, device=device)
    g_comp[:, 2] = env._gravity_magnitude
    a_des = cfg.Kv * (v_des_w - v_curr_w) + g_comp  # (N, 3)

    # Desired force vector
    masses = env._masses.expand(num_envs, num_drones).reshape(N)  # (N,)
    F_des = masses.unsqueeze(-1) * a_des  # (N, 3)

    # Scalar thrust: project desired force onto current body-z axis
    b3_curr = R[:, :, 2]  # current body-z in world frame (N, 3)
    T = (F_des * b3_curr).sum(dim=-1).clamp(min=0.0)  # (N,)

    # Desired body-z from force direction
    F_des_norm = F_des.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    b3_des = F_des / F_des_norm  # (N, 3)

    # Desired heading: integrate yaw command from current yaw angle
    yaw_curr = torch.atan2(R[:, 1, 0], R[:, 0, 0])  # (N,)
    yaw_des = yaw_curr + yaw_rate_des * dt            # (N,)
    b1_c = torch.stack(
        [torch.cos(yaw_des), torch.sin(yaw_des), torch.zeros_like(yaw_des)], dim=-1
    )  # (N, 3)

    # Build desired orthonormal frame (columns of R_des)
    b2_des = torch.linalg.cross(b3_des, b1_c)
    b2_des = b2_des / b2_des.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    b1_des = torch.linalg.cross(b2_des, b3_des)
    R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)  # (N, 3, 3)

    # Attitude error: vee( R_des^T R - R^T R_des ) / 2
    eR_mat = R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des  # (N, 3, 3)
    eR = 0.5 * _vee(eR_mat)  # (N, 3)

    # Angular velocity error (desired body-frame omega = [0, 0, yaw_rate_des])
    Omega_des_b = torch.zeros(N, 3, device=device)
    Omega_des_b[:, 2] = yaw_rate_des
    eOmega = all_omega_b - Omega_des_b  # (N, 3)

    # Inner-loop moment control law
    M = -cfg.KR * eR - cfg.KOmega * eOmega  # (N, 3)

    # Pack into env tensor format (num_envs, num_drones, 1, 3)
    # Clamp to physical limits: max thrust = thrust_to_weight * weight;
    # max moment matches the direct controller's moment_scale * 1 (unit action).
    # _robot_weights shape: (num_drones,) → expand to (num_envs, num_drones) → (N,)
    max_thrust_N = (env._robot_weights * env.cfg.thrust_to_weight).unsqueeze(0).expand(num_envs, num_drones).reshape(N)
    T_clamped = torch.minimum(T.clamp(min=0.0), max_thrust_N)

    max_moment_Nm = env.cfg.moment_scale  # keep same ceiling as direct controller
    M_clamped = M.clamp(-max_moment_Nm, max_moment_Nm)

    thrust = torch.zeros(num_envs, num_drones, 1, 3, device=device)
    thrust[:, :, 0, 2] = T_clamped.reshape(num_envs, num_drones)
    moment = M_clamped.reshape(num_envs, num_drones, 3).unsqueeze(2)

    thrust = torch.nan_to_num(thrust, nan=0.0, posinf=0.0, neginf=0.0)
    moment = torch.nan_to_num(moment, nan=0.0, posinf=0.0, neginf=0.0)

    return thrust, moment


def compute_pd_velocity_controller(
    env,
    actions_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simplified PD velocity controller (near-hover approximation).

    Faster to compute but less accurate at large tilt angles. Suitable as a
    lightweight fallback or for initial debugging.

    Args:
        env: ``BaseSwarmEnv`` instance.
        actions_tensor: ``(num_envs, num_drones, 4)`` ∈ [-1, 1].

    Returns:
        thrust: ``(num_envs, num_drones, 1, 3)``
        moment: ``(num_envs, num_drones, 1, 3)``
    """
    cfg = env.cfg.controller
    num_envs, num_drones = env.num_envs, env.num_drones
    device = env.device

    thrust = torch.zeros(num_envs, num_drones, 1, 3, device=device)
    moment = torch.zeros(num_envs, num_drones, 1, 3, device=device)

    for j, rob in enumerate(env._robots):
        vel_b = rob.data.root_lin_vel_b    # (num_envs, 3)
        omega_b = rob.data.root_ang_vel_b  # (num_envs, 3)
        acts = actions_tensor[:, j, :]

        v_des_b = acts[:, :3] * cfg.max_lin_vel_cmd      # (num_envs, 3)
        yaw_rate_des = acts[:, 3] * cfg.max_yaw_rate_cmd  # (num_envs,)
        vel_err = v_des_b - vel_b                          # (num_envs, 3)

        # Thrust: gravity compensation + altitude velocity P-gain
        T = env._robot_weights[j] * (1.0 + cfg.kp_vel_z * vel_err[:, 2])
        thrust[:, j, 0, 2] = T.clamp(min=0.0)

        # Moments: roll → vy, pitch → -vx, yaw → yaw_rate
        moment[:, j, 0, 0] = cfg.kp_vel_xy * vel_err[:, 1]
        moment[:, j, 0, 1] = -cfg.kp_vel_xy * vel_err[:, 0]
        moment[:, j, 0, 2] = cfg.kp_yaw_rate * (yaw_rate_des - omega_b[:, 2])

    return thrust, moment


def compute_direct_controller(
    env,
    actions_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original direct force/torque control mapping.

    Preserves the exact pre-controller behavior for backward compatibility.

    Args:
        env: ``BaseSwarmEnv`` instance.
        actions_tensor: ``(num_envs, num_drones, 4)`` ∈ [-1, 1].

    Returns:
        thrust: ``(num_envs, num_drones, 1, 3)``
        moment: ``(num_envs, num_drones, 1, 3)``
    """
    num_envs, num_drones = env.num_envs, env.num_drones
    device = env.device

    thrust = torch.zeros(num_envs, num_drones, 1, 3, device=device)
    moment = torch.zeros(num_envs, num_drones, 1, 3, device=device)

    for j in range(num_drones):
        thrust_cmd = (actions_tensor[:, j, 0] + 1.0) / 2.0
        thrust[:, j, 0, 2] = env.cfg.thrust_to_weight * env._robot_weights[j] * thrust_cmd
        moment[:, j, 0, :] = env.cfg.moment_scale * actions_tensor[:, j, 1:]

    return thrust, moment


# ---------------------------------------------------------------------------
# Residual-RL baseline
# ---------------------------------------------------------------------------

def compute_baseline_action(env, kp: float) -> torch.Tensor:
    """P-controller reference command: body-frame velocity proportional to the position
    error toward env._desired_pos_w, no yaw term. Same control law proven in
    scripts/torchrl/flight_test.py's --goto_target mode (converges ~4.85m -> ~0.1-0.2m in
    ~200 steps with zero training), generalized to read the env's own live goal buffer.
    Used as the residual-RL baseline: final action = baseline + residual_scale * policy.

    Returns actions_tensor of shape (num_envs, num_drones, 4) in [-1, 1], the same layout
    apply_controller expects.
    """
    cfg = env.cfg.controller
    all_positions = torch.stack([rob.data.root_pos_w for rob in env._robots], dim=0)  # (D, E, 3)
    all_quats = torch.stack([rob.data.root_quat_w for rob in env._robots], dim=0)      # (D, E, 4)
    desired = env._desired_pos_w.transpose(0, 1)  # (D, E, 3)
    error_w = desired - all_positions
    error_b = quat_apply_inverse(
        all_quats.reshape(-1, 4), error_w.reshape(-1, 3)
    ).reshape(env.num_drones, env.num_envs, 3)
    vel_cmd_b = (kp * error_b / cfg.max_lin_vel_cmd).clamp(-1.0, 1.0)
    baseline = torch.zeros(env.num_drones, env.num_envs, 4, device=env.device)
    baseline[:, :, :3] = vel_cmd_b
    return baseline.transpose(0, 1)  # (E, D, 4)


# ---------------------------------------------------------------------------
# Dispatch entry point
# ---------------------------------------------------------------------------

def apply_controller(
    env,
    actions_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the controller configured in ``env.cfg.controller.type``.

    Args:
        env: ``BaseSwarmEnv`` instance.
        actions_tensor: ``(num_envs, num_drones, 4)`` ∈ [-1, 1].

    Returns:
        thrust: ``(num_envs, num_drones, 1, 3)``
        moment: ``(num_envs, num_drones, 1, 3)``

    Raises:
        ValueError: If ``env.cfg.controller.type`` is not recognized.
    """
    controller_type = env.cfg.controller.type
    if controller_type == "geometric":
        return compute_geometric_controller(env, actions_tensor)
    elif controller_type == "pd_velocity":
        return compute_pd_velocity_controller(env, actions_tensor)
    elif controller_type == "direct":
        return compute_direct_controller(env, actions_tensor)
    else:
        raise ValueError(
            f"Unknown controller type: '{controller_type}'. "
            "Expected 'geometric', 'pd_velocity', or 'direct'."
        )
