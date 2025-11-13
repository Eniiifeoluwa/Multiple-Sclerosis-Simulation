import streamlit as st
import numpy as np
from src.simulation import simulate_cohort
from src.llm_layer import generate_insights, suggest_genes_for_perturbation, memory
from src.utils import plot_lesion_matrix, plot_kaplan_meier
from io import BytesIO

st.set_page_config(page_title="MS Simulation + ChatGroq", layout="wide")
st.title("🧬 MS Simulation, Kaplan-Meier & ChatGroq Chat")

# --- Inputs ---
num_patients = st.slider("Number of Patients", 5, 50, 20)
duration = st.slider("Simulation Duration (days)", 10, 100, 50)
drug_effectiveness = st.slider("Treatment Strength (0=none, 1=full)", 0.0, 1.0, 0.2, 0.05)
remission_chance = st.slider("Remission Chance per Day", 0.0, 0.5, 0.05, 0.01)
immune_variability = st.slider("Immune Activity Variability", 0.0, 0.5, 0.1, 0.05)

simulate_genes = st.checkbox("Use Initial Gene Perturbations", True)
gene_factors = None
if simulate_genes:
    gene_factors = {
        "GeneA": {"immune": np.random.uniform(-0.1,0.1), "neuron": np.random.uniform(-0.1,0.1)},
        "GeneB": {"immune": np.random.uniform(-0.1,0.1), "neuron": np.random.uniform(-0.1,0.1)},
    }

if st.button("Run Simulation + ChatGroq"):
    times, lesion_matrix = simulate_cohort(
        num_patients=num_patients,
        timesteps=duration,
        drug_effectiveness=drug_effectiveness,
        gene_factors=gene_factors,
        remission_chance=remission_chance,
        immune_variability=immune_variability
    )

    # --- Lesion Trajectories Plot ---
    st.subheader("📊 Lesion Trajectories")
    st.pyplot(plot_lesion_matrix(times, lesion_matrix, max_patients=min(10,num_patients)))

    # --- Kaplan-Meier Plot ---
    st.subheader("📈 Lesion-Free Probability (Kaplan-Meier)")
    st.pyplot(plot_kaplan_meier(lesion_matrix, threshold=5))

    # --- ChatGroq Insights ---
    st.subheader("💬 ChatGroq Insights")
    insights = generate_insights(lesion_matrix, top_genes=list(gene_factors.keys()) if gene_factors else None)
    st.text_area("ChatGroq Output", insights, height=300)

    # --- Suggested Gene Perturbations ---
    suggested_genes = suggest_genes_for_perturbation(lesion_matrix, num_genes=3)
    if suggested_genes:
        st.markdown("**ChatGroq Suggested Gene Perturbations:**")
        for gene,effects in suggested_genes.items():
            st.markdown(f"- **{gene}** → Immune effect: {effects['immune']:.2f}, Neuron effect: {effects['neuron']:.2f}")

# Optional: View memory
if st.checkbox("Show Chat Memory"):
    st.json(memory.chat_memory.messages)
