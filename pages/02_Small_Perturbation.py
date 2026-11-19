"""Page 02 - small perturbation dynamics of the linearized model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import astuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import solve_ivp

import dp_model as dp
import dp_plot as dpp
from dp_ui import param_sidebar, pole_table


@st.cache_data(show_spinner=False)
def _response_gif(p_tuple, z0_tuple, t_end_, n_frames):
    """Animated GIF of the nonlinear free response (cached per parameters)."""
    p = dp.Params(*p_tuple)
    t, y = dp.simulate(p, np.array(z0_tuple), t_end_, n=1000)
    return dpp.pendulum_gif_bytes(p, t, y, fps=30, max_frames=n_frames)

st.title("Small perturbation dynamics")
params = param_sidebar()

with st.sidebar:
    st.subheader("Initial perturbation")
    dth1 = st.slider(r"$\theta_1(0)$ [deg]", -10.0, 10.0, 2.0, 0.1, key="p2_th1")
    dth2 = st.slider(r"$\theta_2(0)$ [deg]", -10.0, 10.0, -1.0, 0.1, key="p2_th2")
    dth1d = st.slider(r"$\dot\theta_1(0)$ [deg/s]", -60.0, 60.0, 0.0, 1.0, key="p2_th1d")
    dth2d = st.slider(r"$\dot\theta_2(0)$ [deg/s]", -60.0, 60.0, 0.0, 1.0, key="p2_th2d")
    t_end = st.slider("Simulation time [s]", 2.0, 12.0, 8.0, 0.5, key="p2_tend")
    show_nonlin = st.checkbox("Overlay nonlinear plant", value=True, key="p2_nonlin")
    show_gif = st.checkbox("Show animation", value=False, key="p2_gif")

st.markdown(r"""
## Free response of the linearized system

For a small displacement from the upright pose and no control force, the
dynamics are $\dot z = A z$ with $z = (\delta q, \delta\dot q)$. The exact
solution is $z(t) = e^{At} z_0$, a superposition of the six eigenmodes. Two
modes are the free translation of the cart ($x$ and $\dot x$); the four
remaining modes come in two real pairs $\lambda = \pm a$ and $\lambda = \pm b$,
so each angle is a sum of growing and decaying exponentials. The growing mode
dominates after a few time constants and the tilt increases exponentially.
""")

A, B, M0, Gprime = dp.linearize(params)
eigs, V = np.linalg.eig(A)
order = np.argsort(-eigs.real)
eigs = eigs[order]
V = V[:, order]

lam_u = eigs[eigs.real > 1e-9].real.max()
tau = 1.0 / lam_u
t_double = np.log(2.0) / lam_u

m1, m2, m3, m4 = st.columns(4)
m1.metric("Dominant unstable pole", f"{lam_u:+.3f} 1/s")
m2.metric("Time constant", f"{tau:.3f} s")
m3.metric("Doubling time", f"{t_double:.3f} s")
m4.metric("Stable pole", f"{eigs[eigs.real < -1e-9].real.min():+.3f} 1/s")

z0 = np.array([0.0, np.deg2rad(dth1), np.deg2rad(dth2),
               0.0, np.deg2rad(dth1d), np.deg2rad(dth2d)])
t_eval = np.linspace(0.0, t_end, 1000)
sol = solve_ivp(lambda t, z: A @ z, (0.0, t_end), z0, t_eval=t_eval,
                method="DOP853", rtol=1e-10, atol=1e-12)
t_lin, y_lin = sol.t, sol.y

if show_nonlin:
    t_nl, y_nl = dp.simulate(params, z0, t_end, n=1000)
else:
    t_nl = y_nl = None

if show_gif:
    if y_nl is None:
        t_nl, y_nl = dp.simulate(params, z0, t_end, n=1000)
    with st.spinner("Rendering animation..."):
        gif = _response_gif(astuple(params), tuple(z0), t_end, 120)
    st.image(gif, caption="Animation of the nonlinear response to the perturbation")

# ---------------------------------------------------------------- angles
fig, ax = dpp.new_fig(figsize=(8, 3.8))
a = ax[0, 0]
a.plot(t_lin, np.rad2deg(y_lin[1]), color=dpp.C_T1, lw=2, label=r"$\theta_1$ linear")
a.plot(t_lin, np.rad2deg(y_lin[2]), color=dpp.C_T2, lw=2, label=r"$\theta_2$ linear")
if show_nonlin:
    a.plot(t_nl, np.rad2deg(y_nl[1]), color=dpp.C_T1, lw=1.2, ls="--", label=r"$\theta_1$ nonlinear")
    a.plot(t_nl, np.rad2deg(y_nl[2]), color=dpp.C_T2, lw=1.2, ls="--", label=r"$\theta_2$ nonlinear")
a.set_xlabel("t [s]")
a.set_ylabel("angle [deg]")
a.set_title("Angle response to the initial perturbation")
a.legend(fontsize=8, ncol=2)
st.pyplot(fig)

# ------------------------------------------------------- exponential growth
fig, ax = dpp.new_fig(figsize=(8, 3.4))
a = ax[0, 0]
mask = np.abs(y_lin[1]) > np.deg2rad(0.05)
a.semilogy(t_lin, np.abs(np.rad2deg(y_lin[1])), color=dpp.C_T1, lw=1.8, label=r"$|\theta_1|$")
a.semilogy(t_lin, np.abs(np.rad2deg(y_lin[2])), color=dpp.C_T2, lw=1.8, label=r"$|\theta_2|$")
fit_text = "fit not available"
if mask.sum() > 30:
    slope, intercept = np.polyfit(t_lin[mask], np.log(np.abs(np.rad2deg(y_lin[1])[mask])), 1)
    a.semilogy(t_lin, np.exp(intercept + slope * t_lin), "k--", lw=1.0,
               label=fr"fit: slope ${slope:.3f}\ \mathrm{{s^{{-1}}}}$")
    fit_text = f"{slope:.3f} 1/s"
a.set_xlabel("t [s]")
a.set_ylabel("|angle| [deg]")
a.set_title("Exponential growth of the unstable mode")
a.legend(fontsize=8)
st.pyplot(fig)
st.caption(
    "The fitted slope is the growth rate of the dominant mode and should match "
    f"the dominant pole {lam_u:.3f} 1/s. Fitted slope: {fit_text}."
)

# ------------------------------------------------------------- cart motion
fig, ax = dpp.new_fig(2, 1, figsize=(8, 5.2))
ax[0, 0].plot(t_lin, y_lin[0], color=dpp.C_X, lw=2)
ax[0, 0].set_ylabel("x [m]")
ax[0, 0].set_title("Cart position and velocity (linear model)")
ax[1, 0].plot(t_lin, y_lin[3], color=dpp.C_X, lw=2)
ax[1, 0].set_ylabel(r"$\dot x$ [m/s]")
ax[1, 0].set_xlabel("t [s]")
st.pyplot(fig)

# ------------------------------------------------------- modal decomposition
st.markdown(r"""
## Modal decomposition of the response

Writing $z(t) = \sum_k c_k e^{\lambda_k t} v_k$ with $c = V^{-1} z_0$, the
table lists every pole, its mode shape in the angular coordinates, and the
coefficient excited by the chosen initial condition. The unstable modes carry
the growth; the stable modes die out with the rate $|\mathrm{Re}\,\lambda|$.
""")

c = np.linalg.solve(V, z0)
rows = []
for k, lam in enumerate(eigs):
    if abs(lam) < 1e-9:
        shape = "x translation"
        ck = ""
    else:
        v = V[:, k]
        shape = f"theta1 = {v[1]:+.3f}, theta2 = {v[2]:+.3f}"
        ck = f"{c[k]:+.4f}"
    rows.append({
        "pole": f"{lam.real:+.4f}{lam.imag:+.4f}j",
        "type": ("zero" if abs(lam) < 1e-9 else ("unstable" if lam.real > 0 else "stable")),
        "mode shape": shape,
        "coefficient c_k": ck,
    })
df = pd.DataFrame(rows)
df.index.name = "mode"
st.dataframe(df, width="stretch")

# ------------------------------------------------------------ phase portrait
st.markdown(r"""
## Phase portrait

Trajectory in the $(\theta_1, \dot\theta_1)$ plane. The straight lines are the
stable and unstable eigendirections of $A$ projected onto that plane: the
motion is attracted to the unstable direction and then driven away along it.
""")
fig, ax = dpp.new_fig(figsize=(6.5, 5.0))
a = ax[0, 0]
if show_nonlin:
    a.plot(np.rad2deg(y_nl[1]), np.rad2deg(y_nl[4]), color="0.3", lw=1.2,
           label="nonlinear trajectory")
a.plot(np.rad2deg(y_lin[1]), np.rad2deg(y_lin[4]), color=dpp.C_T1, lw=1.8,
       label="linear trajectory")
for lam, v in zip(eigs, V.T):
    if abs(lam) < 1e-9:
        continue
    direction = np.array([v[1], v[4]])
    direction = direction / np.linalg.norm(direction)
    col = dpp.C_T1 if lam.real > 0 else dpp.C_T2
    a.plot([-40 * direction[0], 40 * direction[0]],
           [-40 * direction[1], 40 * direction[1]], color=col, lw=1.0, ls=":")
a.set_xlabel(r"$\theta_1$ [deg]")
a.set_ylabel(r"$\dot\theta_1$ [deg/s]")
a.set_title("Phase plane")
a.legend(fontsize=8)
st.pyplot(fig)

# ----------------------------------------------------------- energy check
if show_nonlin:
    E = np.array([dp.total_energy(y_nl[:, i], params) for i in range(y_nl.shape[1])])
    drift = np.max(np.abs(E - E[0])) / abs(E[0])
    st.markdown(r"""
## Energy check

With no actuator force the free system conserves total energy. The relative
drift over the simulation is
""")
    st.metric("Relative energy drift", f"{drift:.2e}")
    fig, ax = dpp.new_fig(figsize=(8, 3.0))
    a = ax[0, 0]
    a.plot(t_nl, E, color=dpp.C_U, lw=1.8, label="total energy")
    a.plot(t_nl, [dp.potential_energy(y_nl[:, i], params) for i in range(y_nl.shape[1])],
           color=dpp.C_T1, lw=1.2, ls="--", label="potential")
    a.plot(t_nl, [dp.kinetic_energy(y_nl[:, i], params) for i in range(y_nl.shape[1])],
           color=dpp.C_T2, lw=1.2, ls="--", label="kinetic")
    a.set_xlabel("t [s]")
    a.set_ylabel("energy [J]")
    a.set_title("Energy of the free nonlinear plant")
    a.legend(fontsize=8)
    st.pyplot(fig)
