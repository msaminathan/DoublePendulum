"""Shared Streamlit UI helpers (parameter sidebar, formatting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import dp_model as dp


def param_sidebar() -> dp.Params:
    """Physical-parameter sliders shared by every page."""
    with st.sidebar:
        st.subheader("Physical parameters")
        M = st.slider("Cart mass  M  [kg]", 0.2, 5.0, 1.0, 0.1, key="dp_M")
        m1 = st.slider("Lower mass  m1  [kg]", 0.05, 2.0, 0.5, 0.05, key="dp_m1")
        m2 = st.slider("Upper mass  m2  [kg]", 0.05, 2.0, 0.3, 0.05, key="dp_m2")
        l1 = st.slider("Lower rod  l1  [m]", 0.10, 1.50, 0.50, 0.05, key="dp_l1")
        l2 = st.slider("Upper rod  l2  [m]", 0.10, 1.50, 0.40, 0.05, key="dp_l2")
        g = st.slider("Gravity  g  [m/s^2]", 1.0, 20.0, 9.81, 0.01, key="dp_g")
    return dp.Params(M=M, m1=m1, m2=m2, l1=l1, l2=l2, g=g)


def pole_table(eigs: np.ndarray, tol: float = 1e-9) -> pd.DataFrame:
    """DataFrame describing the poles of a linear system."""
    rows = []
    for lam in eigs:
        if abs(lam) < tol:
            kind = "zero (translation)"
        elif lam.real > tol:
            kind = "unstable"
        elif lam.real < -tol:
            kind = "stable"
        else:
            kind = "oscillatory"
        rows.append({
            "pole": f"{lam.real:+.4f}{lam.imag:+.4f}j",
            "type": kind,
            "Re(lambda)": lam.real,
            "Im(lambda)": lam.imag,
        })
    df = pd.DataFrame(rows)
    df.index.name = "mode"
    return df


def metric_row(cols, items):
    """Render a row of metrics; items = list of (label, value)."""
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)
