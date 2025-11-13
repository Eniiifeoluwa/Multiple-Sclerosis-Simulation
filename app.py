import streamlit as st
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from src.simulation import simulate_cohort
from src.llm_layer import generate_insights, suggest_genes_for_perturbation, memory

st.set_page_config(page_title="MS Simulation + KM + ChatGroq", layout="wide")
st.title("🧬 MS Simulation, Kaplan-Meier, & ChatGroq Insights")

# --- Layman-friendly inputs ---
num_patients = st.slider("Number of simulated patients", 5, 50, 20)
duration = st.slider("Simulation duration (days)", 10, 100, 50)
treatment_strength = st.slider("Treatment strength (0=none, 1=full)", 0.0, 1.0, 0.2, 0.05)
remission_chance = st.slider("Remission chance per day", 0.0, 0.5, 0.05, 0.01)
immune_variability = st.slider("Immune activity variability", 0.0, 0.5, 0.1, 0.05)

simulate_genes = st.checkbox("Use initial gene perturbations", value=True)
gene_factors = None
if simulate_genes:
    gene_factors = {
        "GeneA": {"immune": np.random.uniform(-0.1,0.1), "neuron": np.random.uniform(-0.1,0.1)},
        "GeneB": {"immune": np.random.uniform(-0.1,0.1), "neuron": np.random.uniform(-0.1,0.1)},
    }

# --- Run Simulation ---
if st.button("Run Simulation"):
    times, lesion_matrix = simulate_cohort(
        num_patients=num_patients,
        timesteps=duration,
        drug_effectiveness=treatment_strength,
        gene_factors=gene_factors,
        remission_chance=remission_chance,
        immune_variability=immune_variability
    )

    # --- Layman-friendly summary of lesion trajectories ---
    st.subheader("📊 Lesion Trajectories Summary")
    summary_text = ""
    for i in range(min(5, num_patients)):
        lesions = lesion_matrix[i]
        summary_text += f"Patient {i+1}: Lesions started at {lesions[0]:.1f}, peaked at {lesions.max():.1f}, ended at {lesions[-1]:.1f}.\n"
    st.text(summary_text)

    # --- Kaplan-Meier Plot ---
    st.subheader("📈 Lesion-Free Probability (Kaplan-Meier)")
    event_observed = np.any(lesion_matrix > 0, axis=1).astype(int)  # 1 if patient ever had lesions
    kmf = KaplanMeierFitter()
    kmf.fit(durations=np.full(num_patients, duration), event_observed=event_observed)
    st.line_chart(pd.DataFrame({
        "Days": np.arange(duration),
        "Probability Lesion-Free": kmf.survival_function_.values.flatten()
    }))

    # --- ChatGroq Insights ---
    st.subheader("💬 ChatGroq Insights")
    insights = generate_insights(lesion_matrix, top_genes=list(gene_factors.keys()) if gene_factors else None)
    st.markdown(f"**ChatGroq:** {insights}")

    # --- Suggested gene perturbations ---
    suggested_genes = suggest_genes_for_perturbation(lesion_matrix, num_genes=3)
    if suggested_genes:
        st.markdown("**ChatGroq Suggested Gene Perturbations:**")
        for gene, effects in suggested_genes.items():
            st.markdown(f"- **{gene}** → Immune effect: {effects['immune']:.2f}, Neuron effect: {effects['neuron']:.2f}")
