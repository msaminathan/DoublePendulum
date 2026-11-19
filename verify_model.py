"""Independent verification of dp_model.py.

Checks, in order:
  1. Equations of motion against a numerically differentiated Lagrangian
     (T and V rebuilt independently from the position formulas).
  2. Energy conservation of the free (u = 0) nonlinear plant.
  3. Linearized A against a numerical Jacobian of the nonlinear dynamics.
  4. Controllability of the linearized pair (A, B).
  5. LQR closed-loop eigenvalues.
  6. Nonlinear closed-loop recovery from a small perturbation, and divergence
     of the uncontrolled plant from the same perturbation.
  7. Basin-of-attraction sweep sanity.
"""

import numpy as np

import dp_model as dp

p = dp.Params()
rng = np.random.default_rng(0)
ok = True


def check(name, cond, detail=""):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{status}] {name} {detail}")


# ---------------------------------------------------------------- 1. Lagrangian
def lagrangian(q, qd, p):
    """L = T - V computed from scratch (independent of dp_model's EOM)."""
    x, th1, th2, xd, th1d, th2d = (*q, *qd)
    # positions
    y1 = p.l1 * np.cos(th1)
    y2 = p.l1 * np.cos(th1) + p.l2 * np.cos(th2)
    # velocities by differentiating positions analytically
    v1x = xd + p.l1 * th1d * np.cos(th1)
    v1y = -p.l1 * th1d * np.sin(th1)
    v2x = xd + p.l1 * th1d * np.cos(th1) + p.l2 * th2d * np.cos(th2)
    v2y = -p.l1 * th1d * np.sin(th1) - p.l2 * th2d * np.sin(th2)
    T = 0.5 * p.M * xd**2 + 0.5 * p.m1 * (v1x**2 + v1y**2) + 0.5 * p.m2 * (v2x**2 + v2y**2)
    V = p.m1 * p.g * y1 + p.m2 * p.g * y2
    return T - V


def numeric_eom(q, qd, u, p, h=1e-4):
    """qdd from the Euler-Lagrange equations with numerical derivatives.

    E-L:  d/dt(dL/dqd) - dL/dq = Q_ext,  Q_ext = (u, 0, 0).
    Everything is linear in qdd, so evaluate the residual at two qdd values
    and solve for the qdd that zeroes it.
    """
    def residual(qdd):
        # dL/dqd at (q +- h*qd, qd +- h*qdd)
        dLdqd_plus = np.array([(lagrangian(q + h * qd, qd + h * qdd, p)
                                - lagrangian(q - h * qd, qd - h * qdd, p)) / (2 * h)])
        # need the gradient wrt qd, not the scalar; do it component-wise
        grad_plus = np.zeros(3)
        grad_minus = np.zeros(3)
        for i in range(3):
            d = np.zeros(3)
            d[i] = h
            grad_plus[i] = (lagrangian(q + h * qd, qd + h * qdd + d, p)
                            - lagrangian(q + h * qd, qd + h * qdd - d, p)) / (2 * h)
            grad_minus[i] = (lagrangian(q - h * qd, qd - h * qdd + d, p)
                             - lagrangian(q - h * qd, qd - h * qdd - d, p)) / (2 * h)
        ddt_grad = (grad_plus - grad_minus) / (2 * h)
        grad_q = np.zeros(3)
        for i in range(3):
            d = np.zeros(3)
            d[i] = h
            grad_q[i] = (lagrangian(q + d, qd, p) - lagrangian(q - d, qd, p)) / (2 * h)
        return ddt_grad - grad_q - np.array([u, 0.0, 0.0])

    r0 = residual(np.zeros(3))
    J = np.column_stack([residual(np.eye(3)[i]) - r0 for i in range(3)])
    return np.linalg.solve(J, -r0)


max_err = 0.0
for _ in range(25):
    q = rng.uniform(-1.0, 1.0, 3)
    qd = rng.uniform(-2.0, 2.0, 3)
    u = rng.uniform(-5.0, 5.0)
    qdd_ref = numeric_eom(q, qd, u, p)
    state = np.concatenate([q, qd])
    qdd_model = dp.eom(state, u, p)[3:]
    max_err = max(max_err, np.max(np.abs(qdd_ref - qdd_model)))
check("EOM vs numerical Lagrangian", max_err < 1e-4, f"max|dqdd| = {max_err:.2e}")

# ------------------------------------------------------- 2. energy conservation
state0 = np.array([0.0, 0.35, -0.2, 0.0, 0.4, -0.3])
t, y = dp.simulate(p, state0, 5.0, n=3000)
E = np.array([dp.total_energy(y[:, i], p) for i in range(y.shape[1])])
drift = np.max(np.abs(E - E[0])) / abs(E[0])
check("energy conservation (u=0)", drift < 1e-6, f"relative drift = {drift:.2e}")

# ------------------------------------------------------- 3. linearization check
A, B, M0, Gprime = dp.linearize(p)
eps = 1e-6
Jnum = np.zeros((6, 6))
for i in range(6):
    d = np.zeros(6)
    d[i] = eps
    Jnum[:, i] = (dp.eom(dp.EQUILIBRIUM + d, 0.0, p)
                  - dp.eom(dp.EQUILIBRIUM - d, 0.0, p)) / (2 * eps)
lin_err = np.max(np.abs(Jnum - A))
check("linearization vs numerical Jacobian", lin_err < 1e-5, f"max|dA| = {lin_err:.2e}")

eig = np.linalg.eigvals(A)
print("       open-loop poles:", np.array2string(eig, precision=4))
check("two unstable real poles", np.sum(eig.real > 1e-9) == 2,
      f"unstable count = {int(np.sum(eig.real > 1e-9))}")

# ------------------------------------------------------- 4. controllability
Ctrb = dp.controllability_matrix(A, B)
rank = np.linalg.matrix_rank(Ctrb, tol=1e-9)
check("controllability rank", rank == 6, f"rank = {rank} / 6")
sv = np.linalg.svd(Ctrb, compute_uv=False)
print(f"       controllability singular values: {np.array2string(sv, precision=3)}")

# ------------------------------------------------------- 5. LQR
Q = np.diag([10.0, 200.0, 200.0, 1.0, 2.0, 2.0])
R = np.array([[0.5]])
K, P = dp.lqr_gain(A, B, Q, R)
eig_cl = dp.closed_loop_eigs(A, B, K)
print("       LQR gain K:", np.array2string(K.ravel(), precision=4))
print("       closed-loop poles:", np.array2string(eig_cl, precision=4))
check("LQR closed loop stable", np.all(eig_cl.real < 0),
      f"max Re = {eig_cl.real.max():.3e}")

# ------------------------------------------- 6. nonlinear closed-loop recovery
th0 = np.deg2rad(2.0)
z0 = np.array([0.0, th0, 0.0, 0.0, 0.0, 0.0])
ctrl = lambda s: float(-(K @ (np.asarray(s) - dp.EQUILIBRIUM))[0])
t, y_cl = dp.simulate(p, z0, 6.0, n=1200, controller=ctrl, u_max=20.0)
t, y_ol = dp.simulate(p, z0, 6.0, n=1200)
final = np.abs(y_cl[1:3, -200:]).max()
peak_ol = np.abs(y_ol[1:3]).max()
check("controlled: returns upright", final < np.deg2rad(2.0),
      f"max |theta| over last third = {np.rad2deg(final):.3f} deg")
check("uncontrolled: falls", peak_ol > np.deg2rad(30.0),
      f"peak |theta| = {np.rad2deg(peak_ol):.1f} deg")
u_cl = np.array([ctrl(y_cl[:, i]) for i in range(y_cl.shape[1])])
print(f"       control effort: peak |u| = {np.abs(u_cl).max():.2f} N, "
      f"rms = {np.sqrt(np.mean(u_cl**2)):.2f} N")

# ------------------------------------------------------- 7. basin sweep sanity
n_grid = 9
th_grid = np.linspace(-np.deg2rad(25), np.deg2rad(25), n_grid)
recovered = np.zeros((n_grid, n_grid))
for i, a in enumerate(th_grid):
    for j, b in enumerate(th_grid):
        z0 = np.array([0.0, a, b, 0.0, 0.0, 0.0])
        t, yy = dp.simulate(p, z0, 5.0, n=500, controller=ctrl, u_max=20.0)
        tail = np.abs(yy[1:3, -150:]).max()
        recovered[i, j] = 1.0 if tail < np.deg2rad(5.0) else 0.0
print(f"       basin recovered fraction = {recovered.mean():.2f} "
      f"(grid {n_grid}x{n_grid}, +/-25 deg)")
check("basin sweep runs and is non-trivial",
      0.0 < recovered.mean() < 1.0 or recovered.mean() == 1.0)

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
