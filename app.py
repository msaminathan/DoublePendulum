"""Inverted double pendulum on a frictionless cart — multipage Streamlit app.

Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Inverted Double Pendulum on a Cart",
    page_icon="🎠",
    layout="wide",
)

pages = [
    st.Page("pages/01_Mathematical_Background.py",
            title="Mathematical Background", icon="📐"),
    st.Page("pages/02_Small_Perturbation.py",
            title="Small Perturbation Dynamics", icon="📈"),
    st.Page("pages/03_Nonlinear_Simulation.py",
            title="Nonlinear Simulation", icon="🌀"),
    st.Page("pages/04_Control_LQR.py",
            title="Control: Stabilising the Upright Pose", icon="🎯"),
    st.Page("pages/05_Code_Walkthrough.py",
            title="Code Walkthrough", icon="💻"),
    st.Page("pages/06_Animated_Dynamics.py",
            title="Animated Dynamics", icon="🎬"),
]

# App title above the page list. Streamlit's built-in sidebar navigation
# cannot show a heading above itself, so the list is drawn explicitly.
st.sidebar.title("Inverted Double Pendulum on a Cart")
for page in pages:
    st.sidebar.page_link(page, label=page.title, icon=page.icon)

nav = st.navigation(pages, position="hidden")
nav.run()
