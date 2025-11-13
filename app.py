import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from src.simulation import simulate_cohort
from src.utils import plot_lesion_matrix, plot_kaplan_meier
from src.llm_layer import generate_insights, suggest_genes_for_perturbation

st.set_page_config(page_title="Synthovion: MS Simulation + ChatGroq", layout="wide")

st.title("🧬 Synthovion: Multiple Sclerosis Simulation & ChatGroq Insights")

# --- Sidebar for simulation parameters ---
st.sidebar.header("Simulation Settings")
num_patients = st.sidebar.slider("Number of patients", min_value=10, max_value=100, value=20, step=5)
timesteps = st.sidebar.slider("Time steps", min_value=10, max_value=100, value=30, step=5)
drug_effectiveness = st.sidebar.slider("Drug Effectiveness", min_value=0.0, max_value=1.0, value=0.5)
gene_factors = st.sidebar.text_area("Hypothetical gene effects (comma-separated)", value="GeneA,GeneB,GeneC")

# --- Run Simulation ---
if st.button("Run Simulation + ChatGroq"):
    # Simulate cohort
    times, lesion_matrix = simulate_cohort(
        num_patients=num_patients,
        timesteps=timesteps,
        drug_effectiveness=drug_effectiveness,
        gene_factors=gene_factors.split(","),
    )

    # --- Tabs ---
    tabs = st.tabs(["📈 Lesion Trajectories", "🩺 Kaplan-Meier", "💬 ChatGroq Insights", "🎲 Random Predictions"])

    # --- Tab 1: Lesion trajectories ---
    with tabs[0]:
        st.subheader("MS Lesion Trajectories")
        plot_lesion_matrix(times, lesion_matrix, max_patients=min(num_patients, 10))
        st.caption("Each line represents a simulated patient's lesion count over time.")

    # --- Tab 2: Kaplan-Meier ---
    with tabs[1]:
        st.subheader("Kaplan-Meier: Lesion-Free Survival")
        plot_kaplan_meier(lesion_matrix, threshold=5)
        st.caption("Probability of remaining below threshold lesions over time.")

    # --- Tab 3: ChatGroq insights ---
    with tabs[2]:
        st.subheader("ChatGroq Insights")
        try:
            insights = generate_insights(lesion_matrix, top_genes=gene_factors.split(","))
            st.text_area("ChatGroq Output", insights, height=300)

            perturbations = suggest_genes_for_perturbation(lesion_matrix, num_genes=3)
            st.subheader("Suggested Gene Perturbations")
            st.json(perturbations)
        except Exception as e:
            st.error(f"ChatGroq failed: {e}")

    # --- Tab 4: Random Predictions ---
    with tabs[3]:
        st.subheader("Randomized New Patient Simulation")
        n_new = st.slider("Number of new patients", min_value=1, max_value=20, value=5)
        times_new, lesion_matrix_new = simulate_cohort(
            num_patients=n_new,
            timesteps=timesteps,
            drug_effectiveness=drug_effectiveness,
            gene_factors=gene_factors.split(","),
        )
        plot_lesion_matrix(times_new, lesion_matrix_new, max_patients=n_new)
