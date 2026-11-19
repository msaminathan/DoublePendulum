"""Page 06 - animated dynamics: watch the mechanism move for four scenarios."""

import sys
from dataclasses import astuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

import dp_model as dp
import dp_plot as dpp
from dp_ui import param_sidebar

st.title("Animated dynamics")
params = param_sidebar()

with st.sidebar:
    st.subheader("Scenario")
    scenario = st.selectbox(
        "Scenario",
        ["Free fall (no control)",
         "Small perturbation (no control)",
         "LQR balancing",
         "Disturbance rejection (LQR)"],
        key="p6_scenario",
    )
    if scenario == "Small perturbation (no control)":
        th1_0 = st.slider(r"perturbation $\theta_1(0)$ [deg]", 0.1, 5.0, 1.0, 0.1,
                          key="p6_th1")
        th2_0 = st.slider(r"perturbation $\theta_2(0)$ [deg]", -5.0, 5.0, 0.5, 0.1,
                          key="p6_th2")
    elif scenario == "LQR balancing":
        th1_0 = st.slider(r"$\theta_1(0)$ [deg]", 0.0, 20.0, 8.0, 0.1, key="p6_th1")
        th2_0 = st.slider(r"$\theta_2(0)$ [deg]", -15.0, 15.0, -4.0, 0.1, key="p6_th2")
    elif scenario == "Free fall (no control)":
        th1_0 = st.slider(r"$\theta_1(0)$ [deg]", -60.0, 60.0, 30.0, 0.5, key="p6_th1")
        th2_0 = st.slider(r"$\theta_2(0)$ [deg]", -60.0, 60.0, 20.0, 0.5, key="p6_th2")
    else:
        th1_0 = th2_0 = 0.0
    t_end = st.slider("Duration [s]", 2.0, 15.0, 6.0, 0.5, key="p6_tend")
    u_max = st.slider("Actuator limit [N]", 1.0, 100.0, 20.0, 1.0, key="p6_umax")
    pulse = st.slider("Disturbance pulse [N]", 0.0, 30.0, 10.0, 0.5, key="p6_pulse")
    q_th = st.slider("LQR angle weight", 10.0, 1000.0, 200.0, 10.0, key="p6_qth")
    r_weight = st.slider("LQR effort weight R", 0.01, 10.0, 0.5, 0.01, key="p6_R")

CONTROLLED = scenario in ("LQR balancing", "Disturbance rejection (LQR)")

st.markdown(r"""
## Watching the mechanism move

The animation below integrates the full nonlinear equations
$M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) = B\,F$ and draws the mechanism at
each frame. Use the **snapshot time** slider to scrub to any instant: the
schematic, the angle traces and the readouts all follow the same trajectory.

* **Free fall** — released from rest, no force. The upright pose is unstable,
  so the rods accelerate away from vertical and the cart drifts.
* **Small perturbation** — a one-degree tilt. At first the motion follows the
  linear model exactly and the angle grows exponentially with the dominant
  pole; the nonlinear terms only take over once the tilt is large.
* **LQR balancing** — the state feedback $F = \mathrm{sat}(-K z)$ drives the
  rods back to the upright pose and the cart back to $x = 0$.
* **Disturbance rejection** — the same controller with a short force pulse
  applied to the cart part-way through, to show the recovery.
""")

Q = np.diag([10.0, q_th, q_th, 1.0, 2.0, 2.0])
R = np.array([[r_weight]])
A, B, _, _ = dp.linearize(params)
K, _ = dp.lqr_gain(A, B, Q, R) if CONTROLLED else (None, None)

if scenario == "Small perturbation (no control)":
    z0 = np.array([0.0, np.deg2rad(th1_0), np.deg2rad(th2_0), 0.0, 0.0, 0.0])
    sub = f"theta1(0) = {th1_0:.1f} deg, theta2(0) = {th2_0:.1f} deg, no control"
elif scenario == "Free fall (no control)":
    z0 = np.array([0.0, np.deg2rad(th1_0), np.deg2rad(th2_0), 0.0, 0.0, 0.0])
    sub = f"theta1(0) = {th1_0:.1f} deg, theta2(0) = {th2_0:.1f} deg, no control"
elif scenario == "LQR balancing":
    z0 = np.array([0.0, np.deg2rad(th1_0), np.deg2rad(th2_0), 0.0, 0.0, 0.0])
    sub = f"theta1(0) = {th1_0:.1f} deg, theta2(0) = {th2_0:.1f} deg, LQR"
else:
    z0 = np.zeros(6)
    sub = f"upright start, {pulse:.1f} N pulse at t = 1.0 s, LQR"


def control_law(t_, s_):
    if not CONTROLLED:
        return 0.0
    base = pulse if (pulse > 0 and 1.0 <= t_ <= 1.2 and scenario.startswith("Disturbance")) else 0.0
    return float(np.clip(base - (K @ (np.asarray(s_) - dp.EQUILIBRIUM))[0], -u_max, u_max))


t, y = dp.simulate(params, z0, t_end, n=1200, u=control_law)
u_arr = np.array([control_law(t[i], y[:, i]) for i in range(len(t))])


@st.cache_data(show_spinner=False)
def _gif(p_tuple, z0_tuple, t_end_, scenario_, pulse_, u_max_, q_th_, r_, n_frames):
    p = dp.Params(*p_tuple)
    z0_ = np.array(z0_tuple)
    Q_ = np.diag([10.0, q_th_, q_th_, 1.0, 2.0, 2.0])
    R_ = np.array([[r_]])
    A_, B_, _, _ = dp.linearize(p)
    K_, _ = dp.lqr_gain(A_, B_, Q_, R_) if scenario_ in (
        "LQR balancing", "Disturbance rejection (LQR)") else (None, None)
    controlled = K_ is not None

    def law(t__, s__):
        if not controlled:
            return 0.0
        base = pulse_ if (pulse_ > 0 and 1.0 <= t__ <= 1.2
                          and scenario_.startswith("Disturbance")) else 0.0
        return float(np.clip(base - (K_ @ (np.asarray(s__) - dp.EQUILIBRIUM))[0],
                             -u_max_, u_max_))

    tt, yy = dp.simulate(p, z0_, t_end_, n=1200, u=law)
    return dpp.pendulum_gif_bytes(p, tt, yy, fps=30, max_frames=n_frames)


gif = _gif(astuple(params), tuple(z0), t_end, scenario, pulse, u_max, q_th, r_weight, 120)

t_scrub = st.slider("Snapshot time [s]", 0.0, t_end, 0.0, 0.05, key="p6_scrub")
i_scrub = int(np.argmin(np.abs(t - t_scrub)))

c1, c2 = st.columns([3, 2])
with c1:
    st.image(gif, caption=f"Animation: {sub}")
with c2:
    fig, ax = dpp.new_fig(figsize=(4.2, 4.2))
    dpp.schematic(params, y[:3, i_scrub], ax=ax[0, 0],
                  title=f"t = {t[i_scrub]:.2f} s", center_x=0.0,
                  view_half=dpp.trajectory_view(params, y))
    st.pyplot(fig)
    st.markdown(
        f"**At t = {t[i_scrub]:.2f} s**  |  "
        rf"$\theta_1$ = {np.rad2deg(y[1, i_scrub]):+.2f} deg  |  "
        rf"$\theta_2$ = {np.rad2deg(y[2, i_scrub]):+.2f} deg  |  "
        rf"$x$ = {y[0, i_scrub]:+.3f} m  |  "
        rf"$F$ = {u_arr[i_scrub]:+.2f} N"
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Peak angle", f"{np.rad2deg(np.abs(y[1:3]).max()):.1f} deg")
m2.metric("Final angle", f"{np.rad2deg(np.abs(y[1:3, -1]).max()):.2f} deg")
m3.metric("Peak force", f"{np.abs(u_arr).max():.2f} N")
m4.metric("Cart travel", f"{np.abs(y[0]).max():.2f} m")

fig, ax = dpp.new_fig(2, 1, figsize=(8, 5.4))
ax[0, 0].plot(t, np.rad2deg(y[1]), color=dpp.C_T1, lw=1.8, label=r"$\theta_1$")
ax[0, 0].plot(t, np.rad2deg(y[2]), color=dpp.C_T2, lw=1.8, label=r"$\theta_2$")
ax[0, 0].axvline(t[i_scrub], color="0.3", lw=1.0, ls=":")
ax[0, 0].set_ylabel("angle [deg]")
ax[0, 0].set_title(f"Angle traces - {sub}")
ax[0, 0].legend(fontsize=8, ncol=2)
if CONTROLLED:
    ax[1, 0].plot(t, u_arr, color=dpp.C_U, lw=1.6)
    ax[1, 0].set_ylabel("F [N]")
    ax[1, 0].set_title("Control force")
else:
    E = np.array([dp.total_energy(y[:, i], params) for i in range(y.shape[1])])
    ax[1, 0].plot(t, E, color=dpp.C_U, lw=1.6)
    ax[1, 0].set_ylabel("E [J]")
    ax[1, 0].set_title("Total energy (conserved, F = 0)")
ax[1, 0].axvline(t[i_scrub], color="0.3", lw=1.0, ls=":")
ax[1, 0].set_xlabel("t [s]")
st.pyplot(fig)

st.info(r"""
The dotted vertical line marks the snapshot time chosen with the slider. The
animation and the traces come from the same integration, so the moving picture
and the curves always agree.
""")
