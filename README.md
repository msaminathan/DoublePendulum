# Inverted Double Pendulum on a Frictionless Cart

A multipage Streamlit app that derives, simulates and controls the inverted
double pendulum on a cart.

## Run

```bash
cd ~/DoublePendulum
streamlit run app.py
```

The app uses the system Python packages (`streamlit`, `numpy`, `scipy`,
`matplotlib`, `pandas`); `requirements.txt` lists them.

## Pages

| Page | Content |
| --- | --- |
| Mathematical Background | Coordinates, kinetic/potential energy, Euler-Lagrange derivation, `M(q) qdd + C(q,qd) qd + G(q) = B F`, linearization, open-loop poles, modal solution |
| Small Perturbation Dynamics | Interactive free response to a small tilt, exponential growth rate, cart motion, modal decomposition, phase portrait, energy conservation, optional animation |
| Nonlinear Simulation | Full nonlinear plant, time series, phase planes, energy, configuration snapshots, animation, sensitivity to initial conditions |
| Control: Stabilising the Upright Pose | Controllability, LQR design (Riccati), closed-loop poles, controlled vs uncontrolled response, animated comparison, basin of attraction, saturation and swing-up notes |
| Code Walkthrough | Live source of the physics, simulation, control and plotting functions, each with an explanation |
| Animated Dynamics | Four scenarios (free fall, small perturbation, LQR balancing, disturbance rejection) as an animated mechanism with a time scrubber |

Angles are in radians internally; the UI shows degrees.

## Files

- `app.py` - entry point and page navigation
- `dp_model.py` - physics: mass matrix, Coriolis and gravity terms, equations of motion, energies, linearization, simulation, controllability, LQR
- `dp_plot.py` - matplotlib helpers (schematic, snapshots, animation)
- `dp_ui.py` - shared sidebar controls and formatting helpers
- `pages/` - the four pages
- `verify_model.py` - independent checks of the physics

## Verification

```bash
python verify_model.py
```

Checks performed:

1. The equations of motion against a numerically differentiated Lagrangian
   (independent implementation of `T` and `V`).
2. Energy conservation of the free nonlinear plant.
3. The analytic linearization against a numerical Jacobian of the nonlinear
   dynamics at the upright equilibrium.
4. Controllability (rank of the controllability matrix).
5. LQR closed-loop stability.
6. Nonlinear closed-loop recovery from a small perturbation, and divergence
   without control.
7. Basin-of-attraction sweep.

## Physics summary

With `q = (x, theta1, theta2)` measured from the upward vertical:

```
M(q) qdd + C(q, qd) qd + G(q) = B F,      B = (1, 0, 0)^T
```

Linearizing about `q = 0` gives `z_dot = A z + B_u F` with
`z = (dq, dq_dot)`. The upright pose has two unstable real poles (plus two
zero poles from translation symmetry), and the pair `(A, B_u)` is controllable,
so a single cart force stabilizes the whole system. The LQR law
`F = -K z` with `K = R^-1 B_u^T P` (algebraic Riccati equation) gives a local
balancing controller; its basin of attraction is limited by the actuator
saturation and by the validity of the linear model.
