"""Shared matplotlib helpers for the DoublePendulum Streamlit app."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

import dp_model as dp

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 110,
})

C_X = "#4C72B0"
C_T1 = "#C44E52"
C_T2 = "#55A868"
C_U = "#8172B3"
C_LIN = "#937860"


def new_fig(nrows=1, ncols=1, figsize=(8, 4)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    return fig, ax


def trajectory_view(p: dp.Params, y, margin: float = 1.12) -> float:
    """Half-width of a square frame, centred on the origin, that contains the
    whole mechanism (cart pivot and both bobs) over the entire trajectory."""
    span = p.l1 + p.l2
    xs, ys = [], []
    for k in range(y.shape[1]):
        pivot, p1, p2 = dp.joint_positions(y[:3, k], p)
        for pt in (pivot, p1, p2):
            xs.append(pt[0])
            ys.append(pt[1])
    x_half = max(abs(min(xs)), abs(max(xs))) * margin
    y_half = max(abs(min(ys)), abs(max(ys))) * margin
    return float(max(x_half, y_half, 1.12 * span))


def schematic(p: dp.Params, q=(0.0, 0.0, 0.0), ax=None, title=None, center_x=None,
              full_height=False, view_half=None):
    """Draw the cart, both rods and both bobs for the configuration q.

    center_x fixes the middle of the frame (default: the current cart position,
    so the view follows the cart). Pass center_x=0.0 to keep x = 0 centred.
    full_height widens the frame vertically so the rods are visible when they
    point downwards as well as upwards (default framing suits the upright pose).
    view_half, when given, forces a square frame of that half-width centred on
    (center_x, 0) - use trajectory_view() so the whole motion stays in frame.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    x, th1, th2 = q
    pivot, p1, p2 = dp.joint_positions(q, p)
    cx = float(pivot[0] if center_x is None else center_x)

    span = p.l1 + p.l2
    ax.axhline(0.0, color="0.4", lw=1.5, zorder=1)
    for gx in np.linspace(cx - span, cx + span, 13):
        ax.plot([gx - 0.03, gx + 0.03], [-0.02, -0.08], color="0.4", lw=1)

    cart_w, cart_h = 0.32 * span, 0.10 * span
    cart = patches.Rectangle((x - cart_w / 2, -cart_h), cart_w, cart_h,
                             fc="#DDDDDD", ec="0.25", lw=1.2, zorder=3)
    ax.add_patch(cart)
    for wx in (x - cart_w / 3, x + cart_w / 3):
        wheel = patches.Circle((wx, -cart_h), 0.035 * span, fc="0.3", zorder=4)
        ax.add_patch(wheel)

    ax.plot([pivot[0], p1[0]], [pivot[1], p1[1]], color=C_T1, lw=3, zorder=2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=C_T2, lw=3, zorder=2)
    ax.plot(pivot[0], pivot[1], "o", ms=5, color="0.2", zorder=5)
    ax.plot(p1[0], p1[1], "o", ms=10, color=C_T1, zorder=5)
    ax.plot(p2[0], p2[1], "o", ms=10, color=C_T2, zorder=5)

    ax.annotate(r"$\theta_1$", xy=(pivot[0] + 0.28 * p.l1 * np.sin(th1), 0.28 * p.l1 * np.cos(th1)),
                fontsize=11, color=C_T1)
    ax.annotate(r"$\theta_2$", xy=(p1[0] + 0.30 * p.l2 * np.sin(th2), p1[1] + 0.30 * p.l2 * np.cos(th2)),
                fontsize=11, color=C_T2)

    if view_half is not None:
        ax.set_xlim(cx - view_half, cx + view_half)
        ax.set_ylim(-view_half, view_half)
    elif full_height:
        ax.set_xlim(cx - 1.2 * span, cx + 1.2 * span)
        ax.set_ylim(-1.2 * span, 1.2 * span)
    else:
        ax.set_xlim(cx - 1.15 * span, cx + 1.15 * span)
        ax.set_ylim(-0.45 * span, 1.25 * span)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def snapshot_grid(p: dp.Params, t, y, n=6, width=9.0):
    """n snapshots of the mechanism along a trajectory (whole motion in frame)."""
    idx = np.linspace(0, len(t) - 1, n).astype(int)
    half = trajectory_view(p, y)
    fig, axes = plt.subplots(1, n, figsize=(width, 2.6))
    for k, ax in enumerate(axes):
        i = idx[k]
        schematic(p, y[:3, i], ax=ax, center_x=0.0, view_half=half)
        ax.set_title(f"t = {t[i]:.2f} s", fontsize=9)
        ax.set_xlabel("")
    fig.tight_layout()
    return fig


def pendulum_gif_bytes(p: dp.Params, t, y, fps: int = 30, max_frames: int = 160) -> bytes:
    """Animated GIF of a trajectory, returned as bytes.

    Artists are created once and updated per frame (rather than redrawing the
    whole schematic), which keeps the render time to a few seconds.
    """
    import os
    import tempfile

    from matplotlib.animation import FuncAnimation, PillowWriter

    n_frames = min(max_frames, len(t))
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)

    span = p.l1 + p.l2
    cart_w, cart_h = 0.32 * span, 0.10 * span
    view = trajectory_view(p, y)

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=80)
    ax.set_xlim(-view, view)
    ax.set_ylim(-view, view)
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.set_xlabel("x [m]")
    ax.axhline(0.0, color="0.4", lw=1.5, zorder=1)
    for gx in np.linspace(-view, view, 15):
        ax.plot([gx - 0.03, gx + 0.03], [-0.02, -0.08], color="0.4", lw=1)

    cart = patches.Rectangle((0.0, -cart_h), cart_w, cart_h,
                             fc="#DDDDDD", ec="0.25", lw=1.2, zorder=3)
    ax.add_patch(cart)
    wheel_a = patches.Circle((0.0, -cart_h), 0.035 * span, fc="0.3", zorder=4)
    wheel_b = patches.Circle((0.0, -cart_h), 0.035 * span, fc="0.3", zorder=4)
    ax.add_patch(wheel_a)
    ax.add_patch(wheel_b)

    rod1, = ax.plot([], [], color=C_T1, lw=3, zorder=2)
    rod2, = ax.plot([], [], color=C_T2, lw=3, zorder=2)
    pivot_pt, = ax.plot([], [], "o", ms=5, color="0.2", zorder=5)
    bob1, = ax.plot([], [], "o", ms=10, color=C_T1, zorder=5)
    bob2, = ax.plot([], [], "o", ms=10, color=C_T2, zorder=5)
    title = ax.set_title("", fontsize=10)

    def update(k):
        q = y[:3, idx[k]]
        pivot, p1, p2 = dp.joint_positions(q, p)
        cart.set_xy((q[0] - cart_w / 2, -cart_h))
        wheel_a.center = (q[0] - cart_w / 3, -cart_h)
        wheel_b.center = (q[0] + cart_w / 3, -cart_h)
        rod1.set_data([pivot[0], p1[0]], [pivot[1], p1[1]])
        rod2.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        pivot_pt.set_data([pivot[0]], [pivot[1]])
        bob1.set_data([p1[0]], [p1[1]])
        bob2.set_data([p2[0]], [p2[1]])
        title.set_text(f"t = {t[idx[k]]:.2f} s")
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    fd, path = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        anim.save(path, writer=PillowWriter(fps=fps))
        with open(path, "rb") as fh:
            data = fh.read()
    finally:
        plt.close(fig)
        os.unlink(path)
    return data

