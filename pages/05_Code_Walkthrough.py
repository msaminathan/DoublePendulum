"""Page 05 - code walkthrough: the real source of each piece of the model.

Snippets are extracted with inspect.getsource() at render time, so they always
match the code that is actually running.
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import dp_model as dp
import dp_plot as dpp
from dp_ui import param_sidebar

st.title("Code walkthrough")
param_sidebar()


def src(obj) -> str:
    """Source of a function/class, dedented for display."""
    return inspect.getsource(obj)


st.markdown(r"""
## How the code is organised

Everything that does mathematics lives in plain Python modules with no
Streamlit inside. The pages only build widgets and draw the results. That split
is what makes the model testable: `verify_model.py` imports the same module
the app uses and checks it against an independent numerical derivation.

| File | Responsibility |
| --- | --- |
| `app.py` | entry point; `st.navigation` list of the five pages |
| `dp_model.py` | physics: coordinates, mass matrix, Coriolis and gravity terms, equations of motion, energies, linearization, simulation, controllability, LQR |
| `dp_plot.py` | matplotlib helpers: mechanism schematic, snapshot grid, animated GIF |
| `dp_ui.py` | shared parameter sidebar, pole table formatting |
| `pages/01` - `pages/04` | theory, small-perturbation, nonlinear and control pages |
| `pages/05` | this walkthrough |
| `verify_model.py` | standalone checks of the physics |

Every snippet below is printed from the live module with `inspect.getsource`,
so it cannot go stale relative to the running app.
""")

t_over, t_params, t_eom, t_energy, t_lin, t_sim, t_ctrl, t_plot = st.tabs([
    "Overview",
    "Coordinates",
    "Equations of motion",
    "Energies",
    "Linearization",
    "Simulation",
    "Control",
    "Plotting",
])

# ------------------------------------------------------------------ overview
with t_over:
    st.markdown(r"""
## Data flow

```
sidebar sliders  ->  dp.Params  ->  dp.eom / dp.linearize / dp.lqr_gain
                                          |
                                          v
                              scipy solve_ivp or matrix exponential
                                          |
                                          v
                              dp_plot figure  ->  st.pyplot / st.image
```

A page never derives physics. It reads parameters, calls one function from
`dp_model`, and hands the arrays to a plotting helper. The two functions a
page calls most are `dp.simulate` (nonlinear time integration) and
`dp.linearize` (the small-angle model used for poles and LQR).

**Numerical settings** (kept identical everywhere):

* integration: `solve_ivp(..., method="DOP853", rtol=1e-9, atol=1e-11)`
* linear solves: `numpy.linalg.solve`, Riccati: `scipy.linalg.solve_continuous_are`
* angles: radians internally, degrees only in widget labels and plot axes
""")
    st.code(src(dp.Params), language="python")

# ------------------------------------------------------------------ coordinates
with t_params:
    st.markdown(r"""
## Coordinates and parameters

The generalized coordinate is $q = (x, \theta_1, \theta_2)$ with the rod angles
measured from the **upward vertical**. `Params` is a frozen dataclass, so it is
hashable and can be used directly as an `@st.cache_data` key.
""")
    st.code(src(dp.Params), language="python")
    st.markdown(r"""
`joint_positions` turns a configuration into the three points used for
drawing and for the energy: the pivot on the cart, the lower bob and the upper
bob.
""")
    st.code(src(dp.joint_positions), language="python")

# ---------------------------------------------------------------------- eom
with t_eom:
    st.markdown(r"""
## Equations of motion

The model is the standard robotic form

$$M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) = B\,F, \qquad B = (1,0,0)^{T}$$

`mass_matrix` builds $M(q)$, `coriolis_vector` returns the product
$C(q,\dot q)\dot q$ (the Coriolis matrix itself is never formed), and
`gravity_vector` returns $G(q) = \partial V/\partial q$. `eom` solves the
linear system for $\ddot q$ at each integrator step.
""")
    st.code(src(dp.mass_matrix), language="python")
    st.code(src(dp.coriolis_vector), language="python")
    st.code(src(dp.gravity_vector), language="python")
    st.markdown(r"""
`eom` returns the 6-vector derivative of $z = (q, \dot q)$. `residual` is the
same expression rearranged to zero; it is what `verify_model.py` checks
against a numerically differentiated Lagrangian.
""")
    st.code(src(dp.eom), language="python")
    st.code(src(dp.residual), language="python")

# -------------------------------------------------------------------- energy
with t_energy:
    st.markdown(r"""
## Energies

`T` and `V` are written directly from the bob velocities and heights, so they
are independent of the equations of motion - which is exactly why they make a
good check. With $F = 0$ the free system conserves `total_energy`, and
`verify_model.py` asserts a relative drift below $10^{-6}$ over a trajectory.
""")
    st.code(src(dp.kinetic_energy), language="python")
    st.code(src(dp.potential_energy), language="python")
    st.code(src(dp.total_energy), language="python")

# ---------------------------------------------------------------- linearization
with t_lin:
    st.markdown(r"""
## Linearization

Evaluating the mass matrix and the gravity Jacobian at the upright pose gives
$M_0$ and $G'$, and the state-space form follows:

$$A = \begin{bmatrix} 0 & I_3 \\ -M_0^{-1}G' & 0 \end{bmatrix}, \qquad B_u = \begin{bmatrix} 0 \\ M_0^{-1} B \end{bmatrix}$$

The poles of `A` are the open-loop poles shown on pages 01, 02 and 04. Because
`G'` is negative on the two angular rows, `-M_0^{-1} G'` has two positive
eigenvalues, so the upright pose has two unstable modes.
""")
    st.code(src(dp.linearize), language="python")

# ----------------------------------------------------------------- simulation
with t_sim:
    st.markdown(r"""
## Simulation

`simulate` wraps `solve_ivp`. It accepts either a time-dependent force
`u(t, state)` or a state feedback `controller(state)`, and applies an actuator
limit by clipping the total force. A smooth problem (LQR balancing) needs a
few hundred points; a chaotic large-angle run uses more.
""")
    st.code(src(dp.simulate), language="python")

# --------------------------------------------------------------------- control
with t_ctrl:
    st.markdown(r"""
## Controllability and LQR

`controllability_matrix` builds $[B_u, A B_u, \dots, A^{5} B_u]$; its rank
decides whether the single cart force can move every mode. `lqr_gain` solves
the continuous algebraic Riccati equation and returns
$K = R^{-1} B_u^{T} P$; `closed_loop_eigs` reports the poles of $A - B_u K$.
""")
    st.code(src(dp.controllability_matrix), language="python")
    st.code(src(dp.lqr_gain), language="python")
    st.code(src(dp.closed_loop_eigs), language="python")
    st.markdown(r"""
The controller used by page 04 is a two-liner built from these pieces:

```python
K, _ = dp.lqr_gain(A, B, Q, R)
ctrl = lambda s: float(-(K @ (np.asarray(s) - dp.EQUILIBRIUM))[0])
t, y = dp.simulate(params, z0, t_end, n=1500, u=u_total)  # u_total = pulse + ctrl
```
""")

# --------------------------------------------------------------------- plotting
with t_plot:
    st.markdown(r"""
## Plotting

`schematic` draws the rail, cart, rods and bobs for one configuration and is
reused by the snapshot grid and by the animation. `snapshot_grid` shows the
mechanism at evenly spaced instants; `pendulum_gif_bytes` builds an animated
GIF by updating the artists in place rather than redrawing the whole schematic
each frame (about twice as fast).
""")
    st.code(src(dpp.schematic), language="python")
    st.code(src(dpp.snapshot_grid), language="python")

st.markdown(r"""
## Running the checks yourself

```bash
cd ~/DoublePendulum
python verify_model.py     # physics checks, no Streamlit needed
streamlit run app.py       # the app
```

`verify_model.py` compares the equations of motion against a numerically
differentiated Lagrangian, checks energy conservation, compares the analytic
`A` with a numerical Jacobian of the nonlinear dynamics, and verifies the
controllability rank, the LQR closed-loop poles and the closed-loop recovery.
""")
