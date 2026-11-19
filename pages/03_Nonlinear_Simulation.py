"""Page 03 - full nonlinear simulation of the cart double pendulum."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import astuple

import numpy as np
import streamlit as st

import dp_model as dp
import dp_plot as dpp
from dp_ui import param_sidebar

st.title("Nonlinear simulation")
params = param_sidebar()

with st.sidebar:
    st.subheader("Initial condition")
    th1 = st.slider(r"$\theta_1(0)$ [deg]", -180.0, 180.0, 30.0, 0.5, key="p3_th1")
    th2 = st.slider(r"$\theta_2(0)$ [deg]", -180.0, 180.0, 20.0, 0.5, key="p3_th2")
    th1d = st.slider(r"$\dot\theta_1(0)$ [deg/s]", -360.0, 360.0, 0.0, 5.0, key="p3_th1d")
    th2d = st.slider(r"$\dot\theta_2(0)$ [deg/s]", -360.0, 360.0, 0.0, 5.0, key="p3_th2d")
    x0 = st.slider(r"$x(0)$ [m]", -2.0, 2.0, 0.0, 0.05, key="p3_x0")
    t_end = st.slider("Simulation time [s]", 2.0, 20.0, 8.0, 0.5, key="p3_tend")
    force = st.slider("Constant cart force F [N]", -20.0, 20.0, 0.0, 0.5, key="p3_force")
    show_gif = st.checkbox("Show animation", value=False, key="p3_gif")

st.markdown(r"""
## Full nonlinear plant

This page integrates the complete equations of motion

$$M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) = B\,F$$

with no small-angle approximation. The motion is computed with
`scipy.integrate.solve_ivp` (DOP853, relative tolerance $10^{-9}$). Use the
sliders to set the initial configuration; the pendulum is released from rest
unless angular velocities are given. A constant force $F$ can be applied to
the cart.
""")

state0 = np.array([x0, np.deg2rad(th1), np.deg2rad(th2),
                   0.0, np.deg2rad(th1d), np.deg2rad(th2d)])
t, y = dp.simulate(params, state0, t_end, n=1500, u=lambda tt, ss: force)

peak = np.abs(y[1:3]).max()
fell = peak > np.deg2rad(90.0)
E = np.array([dp.total_energy(y[:, i], params) for i in range(y.shape[1])])
drift = np.max(np.abs(E - E[0])) / max(abs(E[0]), 1e-12)

c1, c2, c3 = st.columns(3)
c1.metric("Peak angle reached", f"{np.rad2deg(peak):.1f} deg")
c2.metric("Outcome", "fell / rotated" if fell else "stayed upright-ish")
c3.metric("Relative energy drift", f"{drift:.2e}")

# ------------------------------------------------------------- time series
fig, ax = dpp.new_fig(3, 1, figsize=(8, 7.2))
ax[0, 0].plot(t, y[0], color=dpp.C_X, lw=1.8)
ax[0, 0].set_ylabel("x [m]")
ax[0, 0].set_title("Cart position")
ax[1, 0].plot(t, np.rad2deg(y[1]), color=dpp.C_T1, lw=1.6, label=r"$\theta_1$")
ax[1, 0].plot(t, np.rad2deg(y[2]), color=dpp.C_T2, lw=1.6, label=r"$\theta_2$")
ax[1, 0].set_ylabel("angle [deg]")
ax[1, 0].set_title("Rod angles")
ax[1, 0].legend(fontsize=8, ncol=2)
ax[2, 0].plot(t, np.rad2deg(y[4]), color=dpp.C_T1, lw=1.4, label=r"$\dot\theta_1$")
ax[2, 0].plot(t, np.rad2deg(y[5]), color=dpp.C_T2, lw=1.4, label=r"$\dot\theta_2$")
ax[2, 0].set_ylabel("angular velocity [deg/s]")
ax[2, 0].set_xlabel("t [s]")
ax[2, 0].set_title("Angular velocities")
ax[2, 0].legend(fontsize=8, ncol=2)
st.pyplot(fig)

# --------------------------------------------------------------- phase plane
c1, c2 = st.columns(2)
with c1:
    fig, ax = dpp.new_fig(figsize=(5.2, 4.2))
    a = ax[0, 0]
    a.plot(np.rad2deg(y[1]), np.rad2deg(y[4]), color=dpp.C_T1, lw=1.4)
    a.set_xlabel(r"$\theta_1$ [deg]")
    a.set_ylabel(r"$\dot\theta_1$ [deg/s]")
    a.set_title(r"Phase plane: lower rod")
    st.pyplot(fig)
with c2:
    fig, ax = dpp.new_fig(figsize=(5.2, 4.2))
    a = ax[0, 0]
    a.plot(np.rad2deg(y[2]), np.rad2deg(y[5]), color=dpp.C_T2, lw=1.4)
    a.set_xlabel(r"$\theta_2$ [deg]")
    a.set_ylabel(r"$\dot\theta_2$ [deg/s]")
    a.set_title(r"Phase plane: upper rod")
    st.pyplot(fig)

# ------------------------------------------------------------- energy
st.markdown(r"""
## Energy

With $F = 0$ the total mechanical energy $E = T + V$ is conserved; the drift
reported above is the numerical tolerance of the integrator, not physics.
With a nonzero constant force the energy changes at the rate $\dot E = F\dot x$.
""")
fig, ax = dpp.new_fig(figsize=(8, 3.2))
a = ax[0, 0]
a.plot(t, E, color=dpp.C_U, lw=1.8, label="total")
a.plot(t, [dp.potential_energy(y[:, i], params) for i in range(y.shape[1])],
       color=dpp.C_T1, lw=1.2, ls="--", label="potential")
a.plot(t, [dp.kinetic_energy(y[:, i], params) for i in range(y.shape[1])],
       color=dpp.C_T2, lw=1.2, ls="--", label="kinetic")
a.set_xlabel("t [s]")
a.set_ylabel("energy [J]")
a.legend(fontsize=8)
a.set_title("Energy of the nonlinear plant")
st.pyplot(fig)

# ------------------------------------------------------------- snapshots
st.markdown("## Configuration along the trajectory")
st.pyplot(dpp.snapshot_grid(params, t, y, n=6))

# ------------------------------------------------------------- sensitivity
st.markdown(r"""
## Sensitivity to the initial condition

A double pendulum is a classic chaotic system: two trajectories that start
arbitrarily close diverge exponentially. The plot shows the angular difference
between the trajectory above and one started with a slightly different
$\theta_1(0)$.
""")
delta = st.slider("Initial difference in " + r"$\theta_1(0)$" + " [deg]",
                  0.001, 1.0, 0.05, 0.001, key="p3_delta")
state0_b = state0.copy()
state0_b[1] += np.deg2rad(delta)
t_b, y_b = dp.simulate(params, state0_b, t_end, n=1500, u=lambda tt, ss: force)
diff = np.abs(np.rad2deg(y[1] - y_b[1]))
fig, ax = dpp.new_fig(figsize=(8, 3.2))
a = ax[0, 0]
a.semilogy(t, np.maximum(diff, 1e-9), color=dpp.C_T1, lw=1.8)
a.set_xlabel("t [s]")
a.set_ylabel(r"$|\theta_1 - \theta_1^{'}|$ [deg]")
a.set_title(f"Divergence from an initial difference of {delta:.3f} deg")
st.pyplot(fig)

# --------------------------------------------------------------------- gif
if show_gif:
    @st.cache_data(show_spinner=False)
    def _gif(p_tuple, s0_tuple, t_end_, force_, n_frames):
        p = dp.Params(*p_tuple)
        s0 = np.array(s0_tuple)
        tt, yy = dp.simulate(p, s0, t_end_, n=1500, u=lambda t_, s_: force_)
        return dpp.pendulum_gif_bytes(p, tt, yy, fps=30, max_frames=n_frames)

    with st.spinner("Rendering animation..."):
        gif = _gif(astuple(params), tuple(state0), t_end, force, 120)
    st.image(gif, caption="Animation of the simulated motion")
