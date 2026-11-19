"""Page 04 - controlling the pendulum: LQR design and closed-loop behaviour."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import astuple

import numpy as np
import streamlit as st

import dp_model as dp
import dp_plot as dpp
from dp_ui import param_sidebar, pole_table, references


@st.cache_data(show_spinner=False)
def _free_gif(p_tuple, z0_tuple, t_end_, n_frames):
    """Animated GIF of the uncontrolled response."""
    p = dp.Params(*p_tuple)
    t, y = dp.simulate(p, np.array(z0_tuple), t_end_, n=1500)
    return dpp.pendulum_gif_bytes(p, t, y, fps=30, max_frames=n_frames)


@st.cache_data(show_spinner=False)
def _controlled_gif(p_tuple, z0_tuple, t_end_, pulse_, u_max_, K_tuple, n_frames):
    """Animated GIF of the LQR-controlled response (with optional pulse)."""
    p = dp.Params(*p_tuple)
    K_ = np.array(K_tuple).reshape(1, 6)

    def law(t, s):
        base = pulse_ if (pulse_ > 0 and 1.0 <= t <= 1.2) else 0.0
        return float(np.clip(base - (K_ @ (np.asarray(s) - dp.EQUILIBRIUM))[0],
                             -u_max_, u_max_))

    t, y = dp.simulate(p, np.array(z0_tuple), t_end_, n=1500, u=law)
    return dpp.pendulum_gif_bytes(p, t, y, fps=30, max_frames=n_frames)

st.title("Control: stabilising the upright pose")
params = param_sidebar()

with st.sidebar:
    st.subheader("LQR weights")
    q_x = st.slider("Q: cart position", 0.0, 100.0, 10.0, 1.0, key="p4_qx")
    q_th1 = st.slider("Q: lower angle", 0.0, 1000.0, 200.0, 10.0, key="p4_qth1")
    q_th2 = st.slider("Q: upper angle", 0.0, 1000.0, 200.0, 10.0, key="p4_qth2")
    q_xd = st.slider("Q: cart velocity", 0.0, 50.0, 1.0, 0.5, key="p4_qxd")
    q_th1d = st.slider("Q: lower rate", 0.0, 50.0, 2.0, 0.5, key="p4_qth1d")
    q_th2d = st.slider("Q: upper rate", 0.0, 50.0, 2.0, 0.5, key="p4_qth2d")
    r_weight = st.slider("R: control effort", 0.01, 10.0, 0.5, 0.01, key="p4_R")
    u_max = st.slider("Actuator limit [N]", 1.0, 100.0, 20.0, 1.0, key="p4_umax")
    th0 = st.slider("Initial tilt " + r"$\theta_1(0)$" + " [deg]", 0.0, 20.0, 2.0, 0.1, key="p4_th0")
    th2_0 = st.slider("Initial tilt " + r"$\theta_2(0)$" + " [deg]", -10.0, 10.0, 0.0, 0.1, key="p4_th2_0")
    t_end = st.slider("Simulation time [s]", 2.0, 15.0, 8.0, 0.5, key="p4_tend")
    pulse = st.slider("Disturbance pulse [N]", 0.0, 20.0, 0.0, 0.5, key="p4_pulse")
    show_gif = st.checkbox("Animate controlled vs uncontrolled", value=False, key="p4_gif")

Q = np.diag([q_x, q_th1, q_th2, q_xd, q_th1d, q_th2d])
R = np.array([[r_weight]])

A, B, M0, Gprime = dp.linearize(params)
eigs_ol = np.linalg.eigvals(A)

st.markdown(r"""
## The control problem

The upright pose $q = 0$ is an equilibrium but an unstable one: the linearized
system $\dot z = A z + B_u F$ has two poles in the right half plane, so an
uncontrolled pendulum falls within a few seconds. The task is to choose the
cart force $F(t)$ so that the pendulum returns to (and stays at) the upright
pose, and so that the cart returns to a chosen position.

A single force acting on the cart can control all three degrees of freedom:
the system is **controllable**. The controllability matrix
$\mathcal{C} = [B_u, A B_u, A^{2} B_u, \dots, A^{5} B_u]$ has rank
""")

Ctrb = dp.controllability_matrix(A, B)
rank = np.linalg.matrix_rank(Ctrb, tol=1e-9)
sv = np.linalg.svd(Ctrb, compute_uv=False)
c1, c2, c3 = st.columns(3)
c1.metric("rank(C)", f"{rank} / 6")
c2.metric("smallest singular value", f"{sv.min():.3e}")
c3.metric("condition number", f"{sv.max()/sv.min():.2e}")

st.markdown(rf"""
The rank is full, so every mode - including both unstable ones - can be
reached by the actuator. The small singular value ${sv.min():.2e}$ measures how
"expensive" the hardest direction is; it is the direction in which the cart
force produces the least acceleration response.
""")

# ---------------------------------------------------------------------- LQR
st.markdown(r"""
## Linear-quadratic design

Choose $F$ to minimise the quadratic cost

$$J = \int_0^{\infty} \left(z^{T} Q z + R\,F^{2}\right) dt$$

The optimal law is the linear state feedback

$$F = -K\,z, \qquad K = R^{-1} B_u^{T} P$$

where $P$ solves the algebraic Riccati equation

$$A^{T} P + P A - P B_u R^{-1} B_u^{T} P + Q = 0$$

Weights $Q$ penalise deviations from the upright pose (cart position and
velocity, both rod angles and rates); $R$ penalises control effort. Increasing
$Q$ relative to $R$ moves the closed-loop poles further into the left half
plane (faster rejection) at the price of larger forces.
""")

K, P = dp.lqr_gain(A, B, Q, R)
eigs_cl = dp.closed_loop_eigs(A, B, K)

st.markdown(r"""**Feedback gain** $K$ (acting on $z = (x, \theta_1, \theta_2, \dot x, \dot\theta_1, \dot\theta_2)$):""")
st.dataframe(
    {"state": ["x", "theta1", "theta2", "xd", "th1d", "th2d"],
     "gain": [f"{k:+.3f}" for k in K.ravel()]},
    width="stretch",
)

c1, c2 = st.columns([1.2, 1])
with c1:
    st.dataframe(pole_table(eigs_cl), width="stretch")
with c2:
    fig, ax = dpp.new_fig(figsize=(4.4, 3.6))
    a = ax[0, 0]
    a.scatter(eigs_ol.real, eigs_ol.imag, s=70, marker="x", color="0.35",
              label="open loop", zorder=3)
    a.scatter(eigs_cl.real, eigs_cl.imag, s=70, color=dpp.C_T2,
              label="closed loop", zorder=3)
    a.axvline(0.0, color="0.5", lw=1)
    a.set_xlabel("Re(lambda)")
    a.set_ylabel("Im(lambda)")
    a.set_title("Pole movement under LQR")
    a.legend(fontsize=8)
    st.pyplot(fig)

# ------------------------------------------------------- closed-loop response
st.markdown(r"""
## Closed-loop response

The nonlinear plant is simulated with the LQR law $F = \mathrm{sat}(-K z)$,
including the actuator limit. The same initial tilt is shown without control
for comparison.
""")

z0 = np.array([0.0, np.deg2rad(th0), np.deg2rad(th2_0), 0.0, 0.0, 0.0])
ctrl = lambda s: float(-(K @ (np.asarray(s) - dp.EQUILIBRIUM))[0])

def u_total(t_, s_):
    base = 0.0
    if pulse > 0 and 1.0 <= t_ <= 1.2:
        base = pulse
    return float(np.clip(base + ctrl(s_), -u_max, u_max))

t_cl, y_cl = dp.simulate(params, z0, t_end, n=1500, u=u_total)
t_ol, y_ol = dp.simulate(params, z0, t_end, n=1500)
u_cl = np.array([u_total(t_cl[i], y_cl[:, i]) for i in range(y_cl.shape[1])])

fig, ax = dpp.new_fig(3, 1, figsize=(8, 7.4))
ax[0, 0].plot(t_ol, np.rad2deg(y_ol[1]), color="0.5", lw=1.4, ls="--", label="uncontrolled")
ax[0, 0].plot(t_cl, np.rad2deg(y_cl[1]), color=dpp.C_T1, lw=2, label="controlled")
ax[0, 0].set_ylabel(r"$\theta_1$ [deg]")
ax[0, 0].set_title("Lower rod angle")
ax[0, 0].legend(fontsize=8)
ax[1, 0].plot(t_ol, np.rad2deg(y_ol[2]), color="0.5", lw=1.4, ls="--", label="uncontrolled")
ax[1, 0].plot(t_cl, np.rad2deg(y_cl[2]), color=dpp.C_T2, lw=2, label="controlled")
ax[1, 0].set_ylabel(r"$\theta_2$ [deg]")
ax[1, 0].set_title("Upper rod angle")
ax[1, 0].legend(fontsize=8)
ax[2, 0].plot(t_cl, u_cl, color=dpp.C_U, lw=1.6)
ax[2, 0].set_ylabel("F [N]")
ax[2, 0].set_xlabel("t [s]")
ax[2, 0].set_title("Control force on the cart")
st.pyplot(fig)

tail = np.abs(y_cl[1:3, -300:]).max()
peak_u = np.abs(u_cl).max()
rms_u = np.sqrt(np.mean(u_cl**2))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak angle during recovery", f"{np.rad2deg(np.abs(y_cl[1:3]).max()):.2f} deg")
c2.metric("Residual angle (last 1.5 s)", f"{np.rad2deg(tail):.3f} deg")
c3.metric("Peak force", f"{peak_u:.2f} N")
c4.metric("RMS force", f"{rms_u:.2f} N")

if show_gif:
    with st.spinner("Rendering animations..."):
        g_free = _free_gif(astuple(params), tuple(z0), t_end, 120)
        g_ctrl = _controlled_gif(astuple(params), tuple(z0), t_end, pulse, u_max,
                                 tuple(K.ravel()), 120)
    c1, c2 = st.columns(2)
    with c1:
        st.image(g_free, caption="Uncontrolled: the pendulum falls")
    with c2:
        st.image(g_ctrl, caption="Controlled: LQR returns it to the upright pose")

# ------------------------------------------------------------- basin of attraction
st.markdown(r"""
## Basin of attraction

LQR is a **local** controller: the linear model it is based on is only valid
near the upright pose, and the actuator limit caps how hard it can push. The
map below classifies the outcome for a grid of initial tilts
$(\theta_1(0), \theta_2(0))$ with all velocities zero. A point is green when
the closed loop settles back to the upright pose within the simulated time,
red when the pendulum falls.
""")

@st.cache_data(show_spinner=False)
def basin_map(p_tuple, K_tuple, u_max_, t_end_, n_grid):
    p = dp.Params(*p_tuple)
    K_ = np.array(K_tuple).reshape(1, 6)
    ctrl_ = lambda s: float(-(K_ @ (np.asarray(s) - dp.EQUILIBRIUM))[0])
    span = np.deg2rad(25.0)
    grid = np.linspace(-span, span, n_grid)
    rec = np.zeros((n_grid, n_grid))
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            s0 = np.array([0.0, a, b, 0.0, 0.0, 0.0])
            _, yy = dp.simulate(p, s0, t_end_, n=300, controller=ctrl_, u_max=u_max_,
                                rtol=1e-6, atol=1e-8)
            rec[i, j] = 1.0 if np.abs(yy[1:3, -120:]).max() < np.deg2rad(5.0) else 0.0
    return grid, rec

n_grid = 9
with st.spinner("Sweeping initial conditions..."):
    grid, rec = basin_map(astuple(params), tuple(K.ravel()), u_max, 5.0, n_grid)

fig, ax = dpp.new_fig(figsize=(5.6, 4.6))
a = ax[0, 0]
pc = a.pcolormesh(np.rad2deg(grid), np.rad2deg(grid), rec.T, cmap="RdYlGn",
                  vmin=0, vmax=1, shading="auto")
a.plot(np.rad2deg(z0[1]), np.rad2deg(z0[2]), "o", ms=9, mfc="none",
       mec="black", mew=1.5, label="current initial condition")
a.set_xlabel(r"$\theta_1(0)$ [deg]")
a.set_ylabel(r"$\theta_2(0)$ [deg]")
a.set_title("Basin of attraction of the LQR controller")
a.legend(fontsize=8, loc="upper right")
fig.colorbar(pc, ax=a, ticks=[0, 1], label="recovered")
st.pyplot(fig)
st.caption(
    f"Recovered fraction on this grid: {rec.mean()*100:.0f} %. "
    "The basin shrinks as the actuator limit is reduced or the weights are "
    "made softer."
)

# ---------------------------------------------------------------- discussion
st.markdown(r"""
## Why the controller works

The cart force does not act on the rods directly; it acts through the pivot
acceleration. When the lower rod starts to lean to the right, the controller
accelerates the cart to the right. In the cart's accelerating reference frame
the bobs feel a pseudo-force to the left that pushes them back toward the
upright pose. The feedback law

$$F = -K\,z = -k_x x - k_{\theta_1}\theta_1 - k_{\theta_2}\theta_2 - k_{\dot x}\dot x - k_{\dot\theta_1}\dot\theta_1 - k_{\dot\theta_2}\dot\theta_2$$

combines four pieces of information: how far the rods have fallen, how fast
they are falling, and where the cart is and how fast it is moving. The gains
are not chosen by hand: they come from the Riccati equation, which trades off
the penalty on the deviation against the penalty on the force.

Two practical limits matter:

* **Actuator saturation.** With a bounded force, the controller can only
  accelerate the cart so hard, so the basin of attraction above is finite.
* **Large deviations.** The LQR law is designed on the linearization, so it
  assumes $\sin\theta \approx \theta$. Far from the upright pose the
  approximation fails and the controller can lose the pendulum. Recovering a
  pendulum that has fallen far requires a *swing-up* strategy: pump energy
  into the swing by moving the cart back and forth at the natural frequency
  until the pendulum reaches the upright region, then hand over to the LQR
  balancing controller. The energy of the ideal upright pose is
  $V_{\max} = (m_1+m_2) g l_1 + m_2 g l_2$; a swing-up controller drives the
  total energy toward this value.

The next step for a hardware implementation would be to add an integrator on
the cart position (to remove steady-state error under friction) and to
implement the swing-up plus balancing hand-over.
""")

references([
    "K. J. Åström, R. M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*, Princeton University Press, 2008 — reachability, controllability and state feedback.",
    "D. E. Kirk, *Optimal Control Theory: An Introduction*, Prentice-Hall, 1970 (Dover reprint) — the linear-quadratic regulator and the Riccati equation.",
    "G. F. Franklin, J. D. Powell, A. Emami-Naeini, *Feedback Control of Dynamic Systems*, 7th ed., Pearson, 2015 — LQR design, pole placement and the effect of actuator saturation.",
    "M. W. Spong, 'The Swing Up Control Problem for the Acrobot', *IEEE Control Systems Magazine* 15(1), 49-55 (1995), doi:10.1109/37.341864 — underactuated systems and partial feedback linearization for swing-up.",
    "K. J. Åström, K. Furuta, 'Swinging Up a Pendulum by Energy Control', *Automatica* 36(2), 278-285 (2000), doi:10.1016/S0005-1098(99)00140-5 — energy-based swing-up and the hand-over to a balancing controller.",
])
