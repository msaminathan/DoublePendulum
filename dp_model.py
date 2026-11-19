"""Dynamics of an inverted double pendulum on a frictionless cart.

Coordinates (angles measured from the upward vertical, positive counter-clockwise):

    q  = (x, th1, th2)
    x   : horizontal position of the cart            [m]
    th1 : angle of the lower rod from vertical       [rad]
    th2 : angle of the upper rod from vertical       [rad]
    u   : horizontal force applied to the cart       [N]

State vector:

    z = (x, th1, th2, xd, th1d, th2d)

The equations of motion have the standard robotic form

    M(q) qdd + C(q, qd) qd + G(q) = B u,      B = (1, 0, 0)^T

derived here from the Lagrangian L = T - V with point masses m1, m2 on
massless rods of length l1, l2 and a cart of mass M.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    """Physical parameters of the cart - double pendulum system."""

    M: float = 1.0    # cart mass [kg]
    m1: float = 0.5   # lower bob mass [kg]
    m2: float = 0.3   # upper bob mass [kg]
    l1: float = 0.5   # lower rod length [m]
    l2: float = 0.4   # upper rod length [m]
    g: float = 9.81   # gravity [m/s^2]


DEFAULT = Params()


# ----------------------------------------------------------------------------
# Equations of motion
# ----------------------------------------------------------------------------
def mass_matrix(q, p: Params = DEFAULT) -> np.ndarray:
    """Mass / inertia matrix M(q) (3x3)."""
    th1, th2 = q[1], q[2]
    M11 = p.M + p.m1 + p.m2
    M12 = (p.m1 + p.m2) * p.l1 * np.cos(th1)
    M13 = p.m2 * p.l2 * np.cos(th2)
    M22 = (p.m1 + p.m2) * p.l1**2
    M23 = p.m2 * p.l1 * p.l2 * np.cos(th1 - th2)
    M33 = p.m2 * p.l2**2
    return np.array([[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]])


def coriolis_vector(q, qd, p: Params = DEFAULT) -> np.ndarray:
    """Coriolis / centrifugal term C(q, qd) qd (3,)."""
    th1, th2 = q[1], q[2]
    th1d, th2d = qd[1], qd[2]
    c1 = -(p.m1 + p.m2) * p.l1 * np.sin(th1) * th1d**2 - p.m2 * p.l2 * np.sin(th2) * th2d**2
    c2 = p.m2 * p.l1 * p.l2 * np.sin(th1 - th2) * th2d**2
    c3 = -p.m2 * p.l1 * p.l2 * np.sin(th1 - th2) * th1d**2
    return np.array([c1, c2, c3])


def gravity_vector(q, p: Params = DEFAULT) -> np.ndarray:
    """Gravity term G(q) = dV/dq (3,)."""
    th1, th2 = q[1], q[2]
    g1 = 0.0
    g2 = -(p.m1 + p.m2) * p.g * p.l1 * np.sin(th1)
    g3 = -p.m2 * p.g * p.l2 * np.sin(th2)
    return np.array([g1, g2, g3])


def eom(state, u: float = 0.0, p: Params = DEFAULT) -> np.ndarray:
    """State derivative for z = (x, th1, th2, xd, th1d, th2d)."""
    state = np.asarray(state, dtype=float)
    q, qd = state[:3], state[3:]
    Mmat = mass_matrix(q, p)
    rhs = np.array([u, 0.0, 0.0]) - coriolis_vector(q, qd, p) - gravity_vector(q, p)
    qdd = np.linalg.solve(Mmat, rhs)
    return np.concatenate([qd, qdd])


def residual(q, qd, qdd, u: float = 0.0, p: Params = DEFAULT) -> np.ndarray:
    """EOM residual M qdd + C qd + G - B u (should be zero on a true trajectory)."""
    return (
        mass_matrix(q, p) @ np.asarray(qdd)
        + coriolis_vector(q, qd, p)
        + gravity_vector(q, p)
        - np.array([u, 0.0, 0.0])
    )


# ----------------------------------------------------------------------------
# Energies and kinematics
# ----------------------------------------------------------------------------
def joint_positions(q, p: Params = DEFAULT):
    """Cart pivot, lower bob and upper bob positions."""
    x, th1, th2 = q
    pivot = np.array([x, 0.0])
    p1 = np.array([x + p.l1 * np.sin(th1), p.l1 * np.cos(th1)])
    p2 = np.array([x + p.l1 * np.sin(th1) + p.l2 * np.sin(th2),
                   p.l1 * np.cos(th1) + p.l2 * np.cos(th2)])
    return pivot, p1, p2


def kinetic_energy(state, p: Params = DEFAULT) -> float:
    """T = 1/2 M xd^2 + 1/2 m1 |v1|^2 + 1/2 m2 |v2|^2."""
    state = np.asarray(state, dtype=float)
    x, th1, th2, xd, th1d, th2d = state
    v1x = xd + p.l1 * th1d * np.cos(th1)
    v1y = -p.l1 * th1d * np.sin(th1)
    v2x = xd + p.l1 * th1d * np.cos(th1) + p.l2 * th2d * np.cos(th2)
    v2y = -p.l1 * th1d * np.sin(th1) - p.l2 * th2d * np.sin(th2)
    return (0.5 * p.M * xd**2
            + 0.5 * p.m1 * (v1x**2 + v1y**2)
            + 0.5 * p.m2 * (v2x**2 + v2y**2))


def potential_energy(state, p: Params = DEFAULT) -> float:
    """V = m1 g l1 cos(th1) + m2 g (l1 cos(th1) + l2 cos(th2))."""
    _, th1, th2 = np.asarray(state, dtype=float)[:3]
    return p.m1 * p.g * p.l1 * np.cos(th1) + p.m2 * p.g * (p.l1 * np.cos(th1) + p.l2 * np.cos(th2))


def total_energy(state, p: Params = DEFAULT) -> float:
    return kinetic_energy(state, p) + potential_energy(state, p)


# ----------------------------------------------------------------------------
# Linearization about the upright equilibrium q = 0
# ----------------------------------------------------------------------------
def linearize(p: Params = DEFAULT):
    """Linearized model about the upright equilibrium.

    Returns A (6x6), B (6x1), M0 (3x3 mass matrix at q=0) and Gprime = dG/dq|0.
    """
    M0 = mass_matrix(np.zeros(3), p)
    Gprime = np.array([[0.0, 0.0, 0.0],
                       [0.0, -(p.m1 + p.m2) * p.g * p.l1, 0.0],
                       [0.0, 0.0, -p.m2 * p.g * p.l2]])
    Minv = np.linalg.inv(M0)
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = -Minv @ Gprime
    B = np.zeros((6, 1))
    B[3:, 0] = Minv @ np.array([1.0, 0.0, 0.0])
    return A, B, M0, Gprime


EQUILIBRIUM = np.zeros(6)


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate(p: Params, state0, t_end: float, n: int = 2000, u=None,
             controller=None, u_max: float | None = None,
             rtol: float = 1e-9, atol: float = 1e-11):
    """Integrate the nonlinear plant.

    u : callable u(t, state) -> float, or None (free response)
    controller : callable controller(state) -> float, applied as u = controller(state)
    u_max : saturation limit for the applied force
    """
    state0 = np.asarray(state0, dtype=float)

    def _u_of(t, state):
        if controller is not None:
            val = float(np.asarray(controller(state)).ravel()[0])
        elif u is not None:
            val = float(u(t, state))
        else:
            val = 0.0
        if u_max is not None:
            val = float(np.clip(val, -u_max, u_max))
        return val

    def rhs(t, state):
        return eom(state, _u_of(t, state), p)

    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(rhs, (0.0, t_end), state0, t_eval=t_eval,
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol.t, sol.y


# ----------------------------------------------------------------------------
# Control: controllability and LQR
# ----------------------------------------------------------------------------
def controllability_matrix(A, B) -> np.ndarray:
    """[B, AB, A^2 B, ...] for the pair (A, B)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    blocks = []
    Apow = np.eye(n)
    for _ in range(n):
        blocks.append(Apow @ B)
        Apow = Apow @ A
    return np.hstack(blocks)


def lqr_gain(A, B, Q, R):
    """Continuous-time LQR gain K = R^-1 B^T P from the Riccati equation."""
    P = solve_continuous_are(np.asarray(A, float), np.asarray(B, float),
                             np.asarray(Q, float), np.asarray(R, float))
    K = np.linalg.solve(np.asarray(R, float), np.asarray(B, float).T @ P)
    return K, P


def closed_loop_eigs(A, B, K):
    return np.linalg.eigvals(np.asarray(A) - np.asarray(B) @ np.asarray(K))
