"""Page 01 - mathematical background: derivation, linearization, modal solution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp

import dp_model as dp
import dp_plot as dpp
from dp_ui import param_sidebar, pole_table

st.title("Mathematical background")
params = param_sidebar()

# ------------------------------------------------------------------ system
st.markdown(r"""
## The system

A cart of mass $M$ slides without friction on a horizontal rail. Two point
masses $m_1$ (lower bob) and $m_2$ (upper bob) are carried by massless rigid
rods of length $l_1$ and $l_2$. A horizontal force $F(t)$ acts on the cart.
The configuration is described by the generalized coordinate vector

$$q = (x,\ \theta_1,\ \theta_2)$$

where $x$ is the cart position and $\theta_1,\theta_2$ are the rod angles
measured from the **upward vertical**. The upright pose $q = 0$ is an
equilibrium, but an unstable one: the slightest tilt makes the pendulum fall.
""")

fig, ax = dpp.new_fig(figsize=(5.4, 3.6))
dpp.schematic(params, (0.0, np.deg2rad(8.0), np.deg2rad(-6.0)),
              ax=ax[0, 0], title="Inverted double pendulum on a cart")
st.pyplot(fig)

st.markdown(r"""
**Coordinates and kinematics.** With the pivot at $(x,0)$, the two bobs sit at

$$P_1 = \left(x + l_1 \sin\theta_1,\; l_1 \cos\theta_1\right)$$

$$P_2 = \left(x + l_1 \sin\theta_1 + l_2 \sin\theta_2,\; l_1 \cos\theta_1 + l_2 \cos\theta_2\right)$$

so their velocities are

$$v_{1x} = \dot x + l_1 \dot\theta_1 \cos\theta_1, \qquad v_{1y} = -l_1 \dot\theta_1 \sin\theta_1$$

$$v_{2x} = \dot x + l_1 \dot\theta_1 \cos\theta_1 + l_2 \dot\theta_2 \cos\theta_2, \qquad v_{2y} = -l_1 \dot\theta_1 \sin\theta_1 - l_2 \dot\theta_2 \sin\theta_2$$

The upright equilibrium corresponds to $\theta_1 = \theta_2 = 0$ with the cart
at rest; gravity is taken as $g$ in the $+y$ direction being upward, so the
potential energy is highest at the equilibrium.
""")

# --------------------------------------------------------------- energies
st.markdown(r"""
## Kinetic and potential energy

$$T = \tfrac{1}{2} M \dot x^{2} + \tfrac{1}{2} m_1 \lVert v_1 \rVert^{2} + \tfrac{1}{2} m_2 \lVert v_2 \rVert^{2}$$

$$V = m_1 g l_1 \cos\theta_1 + m_2 g \left(l_1 \cos\theta_1 + l_2 \cos\theta_2\right)$$

$$L(q,\dot q) = T - V$$

Because the cart is free to translate, $x$ is a **cyclic** coordinate: it does
not appear in $V$, so the total horizontal momentum is not conserved once a
force $F$ is applied, but the structure $M(q)\ddot q + C(q,\dot q)\dot q + G(q)
= B F$ is the standard one for a mechanical system with one actuator.
""")

# ------------------------------------------------------------ equations
st.markdown(r"""
## Euler-Lagrange equations

With the generalized force $Q = (F, 0, 0)^{T}$,

$$\frac{d}{dt}\frac{\partial L}{\partial \dot q_i} - \frac{\partial L}{\partial q_i} = Q_i, \qquad i = 1,2,3$$

Carrying out the derivatives gives the equations of motion in the compact form

$$M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) = B\,F, \qquad B = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$$

with the symmetric mass matrix

$$M(q) = \begin{bmatrix} M+m_1+m_2 & (m_1+m_2)\,l_1\cos\theta_1 & m_2\,l_2\cos\theta_2 \\ (m_1+m_2)\,l_1\cos\theta_1 & (m_1+m_2)\,l_1^{2} & m_2\,l_1 l_2\cos(\theta_1-\theta_2) \\ m_2\,l_2\cos\theta_2 & m_2\,l_1 l_2\cos(\theta_1-\theta_2) & m_2\,l_2^{2} \end{bmatrix}$$

the Coriolis/centrifugal vector

$$C(q,\dot q)\,\dot q = \begin{bmatrix} -(m_1+m_2)\,l_1\sin\theta_1\,\dot\theta_1^{2} - m_2\,l_2\sin\theta_2\,\dot\theta_2^{2} \\ m_2\,l_1 l_2\sin(\theta_1-\theta_2)\,\dot\theta_2^{2} \\ -m_2\,l_1 l_2\sin(\theta_1-\theta_2)\,\dot\theta_1^{2} \end{bmatrix}$$

and the gravity vector $G(q) = \partial V/\partial q$,

$$G(q) = \begin{bmatrix} 0 \\ -(m_1+m_2)\,g\,l_1\sin\theta_1 \\ -m_2\,g\,l_2\sin\theta_2 \end{bmatrix}$$

The model in `dp_model.py` implements exactly these expressions; the script
`verify_model.py` checks them against a numerically differentiated Lagrangian,
energy conservation, and a numerical Jacobian of the linearization.
""")

# -------------------------------------------------------- linearization
st.markdown(r"""
## Linearization about the upright equilibrium

Write $q = 0 + \delta q$ and $\dot q = 0 + \delta\dot q$. To first order,
$M(q) \to M_0 = M(0)$ and $G(q) \to G'\,\delta q$ with

$$G' = \begin{bmatrix} 0 & 0 & 0 \\ 0 & -(m_1+m_2)\,g\,l_1 & 0 \\ 0 & 0 & -m_2\,g\,l_2 \end{bmatrix}$$

so the linearized dynamics are

$$M_0\,\delta\ddot q + G'\,\delta q = B\,F$$

or, with the state $z = (\delta q, \delta\dot q) \in \mathbb{R}^{6}$,

$$\dot z = A\,z + B_u\,F$$

$$A = \begin{bmatrix} 0 & I_3 \\ -M_0^{-1}G' & 0 \end{bmatrix}, \qquad B_u = \begin{bmatrix} 0 \\ M_0^{-1} B \end{bmatrix}$$

Because $G'$ is negative on the two angular rows, $-M_0^{-1}G'$ has **two
positive eigenvalues**: the upright pose has two unstable modes. The two zero
eigenvalues come from the translation symmetry ($x$ and $\dot x$ are not
restored by anything). The poles for the current parameters are:
""")

A, B, M0, Gprime = dp.linearize(params)
eigs, V = np.linalg.eig(A)
order = np.argsort(-eigs.real)
eigs = eigs[order]
V = V[:, order]

c1, c2 = st.columns([1.2, 1])
with c1:
    st.dataframe(pole_table(eigs), width="stretch")
with c2:
    fig, ax = dpp.new_fig(figsize=(4.2, 3.4))
    a = ax[0, 0]
    a.scatter(eigs.real, eigs.imag, s=70, color=dpp.C_T1, zorder=3)
    for lam in eigs:
        a.annotate(f"{lam.real:+.2f}{lam.imag:+.2f}j", (lam.real, lam.imag),
                   textcoords="offset points", xytext=(6, 5), fontsize=8)
    a.axvline(0.0, color="0.5", lw=1)
    a.axhline(0.0, color="0.5", lw=1)
    a.set_xlabel("Re(lambda)")
    a.set_ylabel("Im(lambda)")
    a.set_title("Open-loop poles (upright equilibrium)")
    st.pyplot(fig)

lam_u = eigs[eigs.real > 1e-9]
i_dom = int(np.argmax(eigs.real))
v_dom = V[:, i_dom].real
ratio_dom = v_dom[2] / v_dom[1] if abs(v_dom[1]) > 1e-12 else np.inf
phase_dom = ("opposite directions (out-of-phase)" if ratio_dom < 0
             else "the same direction (in-phase)")
st.markdown(rf"""
The dominant unstable pole is $\lambda_{{u}} = {lam_u.real.max():.3f}\ \mathrm{{s^{{-1}}}}$,
so a small tilt grows like $e^{{\lambda_u t}}$ with a time constant
$\tau = 1/\lambda_u = {1.0/lam_u.real.max():.3f}\ \mathrm{{s}}$ and a doubling
time $t_{{2}} = \ln 2/\lambda_u = {np.log(2)/lam_u.real.max():.3f}\ \mathrm{{s}}$.
Its eigenvector has $\theta_2/\theta_1 \approx {ratio_dom:+.2f}$, so the two
rods move in {phase_dom} in the fast mode; the slower unstable mode shows the
opposite pattern. Which pattern is faster depends on the mass and length
parameters in the sidebar.
""")

# -------------------------------------------------- small-perturbation solution
st.markdown(r"""
## Small-perturbation solution

For a small initial displacement $z_0$ and no control ($F=0$), the linearized
system is solved exactly by the matrix exponential

$$z(t) = e^{A t}\,z_0 = \sum_{k=1}^{6} c_k\,e^{\lambda_k t}\,v_k, \qquad c = V^{-1} z_0$$

where $\lambda_k, v_k$ are the eigenvalues and eigenvectors of $A$. The two
zero eigenvalues contribute only the free translation of the cart, so each rod
angle is a sum of four exponentials,

$$\theta_i(t) = \sum_{k:\,\lambda_k \neq 0} c_k\,v_{k,i}\,e^{\lambda_k t}$$

For a perturbation that is small enough, the nonlinear plant follows this
linear solution closely. The plot below compares the two for an initial tilt
of $1^\circ$ in each rod (nonlinear simulation dashed, linear solution solid).
""")

th1_0 = np.deg2rad(1.0)
th2_0 = np.deg2rad(1.0)
z0 = np.array([0.0, th1_0, th2_0, 0.0, 0.0, 0.0])
t_end = 6.0
t_eval = np.linspace(0.0, t_end, 800)

sol_lin = solve_ivp(lambda t, z: A @ z, (0.0, t_end), z0, t_eval=t_eval,
                    method="DOP853", rtol=1e-10, atol=1e-12)
t_nl, y_nl = dp.simulate(params, z0, t_end, n=800)

fig, ax = dpp.new_fig(figsize=(8, 3.6))
a = ax[0, 0]
a.plot(sol_lin.t, np.rad2deg(sol_lin.y[1]), color=dpp.C_T1, lw=2, label=r"$\theta_1$ linear")
a.plot(sol_lin.t, np.rad2deg(sol_lin.y[2]), color=dpp.C_T2, lw=2, label=r"$\theta_2$ linear")
a.plot(t_nl, np.rad2deg(y_nl[1]), color=dpp.C_T1, lw=1.2, ls="--", label=r"$\theta_1$ nonlinear")
a.plot(t_nl, np.rad2deg(y_nl[2]), color=dpp.C_T2, lw=1.2, ls="--", label=r"$\theta_2$ nonlinear")
a.set_xlabel("t [s]")
a.set_ylabel("angle [deg]")
a.set_title("Free response to a 1 deg tilt (no control)")
a.legend(fontsize=8, ncol=2)
st.pyplot(fig)

st.info(r"""
The linear and nonlinear curves are indistinguishable for the first seconds:
this is the regime in which the linear model is trustworthy. The angle grows
exponentially, and once it reaches a few tens of degrees the nonlinear terms
($\sin\theta$, $\cos\theta$, centrifugal coupling) take over and the pendulum
falls. The next page studies this small-perturbation regime in detail.
""")
