# app.py
import streamlit as st
import numpy as np
from src.simulation import simulate_cohort
from src.utils import plot_lesion_matrix, plot_kaplan_meier
from src.llm_layer import generate_insights, suggest_genes_for_perturbation

st.set_page_config(page_title="Synthovion", layout="wide")
st.title("🧬 Synthovion: MS Lesion Simulation & Insights")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Simulation Parameters")
num_patients = st.sidebar.slider("Number of Patients", 5, 50, 20)
observation_days = st.sidebar.slider("Observation Days", 10, 50, 20)
drug_effectiveness = st.sidebar.slider("Drug Effectiveness", 0.0, 1.0, 0.5)
gene_input = st.sidebar.text_input("Hypothetical Genes (comma-separated)", "GeneA,GeneB,GeneC")
gene_list = [g.strip() for g in gene_input.split(",") if g.strip()]

# ---------------- Run Simulation ----------------
if st.button("Run Simulation"):

    # Random gene effects
    gene_factors = {
        g: {"immune": np.random.uniform(-0.2, 0.2), "neuron": np.random.uniform(-0.2, 0.2)}
        for g in gene_list
    }

    # Simulate lesions
    days, lesion_matrix = simulate_cohort(
        num_patients=num_patients,
        observation_days=observation_days,
        drug_effectiveness=drug_effectiveness,
        gene_factors=gene_factors
    )

    # Store in session state
    st.session_state.lesion_matrix = lesion_matrix
    st.session_state.days = days
    st.session_state.gene_factors = gene_factors

    # Suggest gene perturbations
    st.session_state.suggested_genes = suggest_genes_for_perturbation(lesion_matrix, gene_list)

# ---------------- Tabs ----------------
if "lesion_matrix" in st.session_state:

    tabs = st.tabs(["Lesion Trajectories", "Kaplan-Meier", "Chat Insights", "Gene Perturbations"])

    # Lesion Trajectories
    with tabs[0]:
        st.subheader("MS Lesion Trajectories")
        st.pyplot(plot_lesion_matrix(st.session_state.days, st.session_state.lesion_matrix))

    # Kaplan-Meier
    with tabs[1]:
        st.subheader("Lesion-Free Survival (Synthetic Kaplan-Meier)")
        st.pyplot(plot_kaplan_meier(st.session_state.lesion_matrix))

    # Chat Insights
    with tabs[2]:
        st.subheader("Simulation Insights Chat")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Ask a question about the simulation:")

        if st.button("Send Question", key="chat_button"):
            if user_input:
                answer = generate_insights(
                    st.session_state.lesion_matrix,
                    top_genes=gene_list,
                    user_question=user_input
                )
                st.session_state.chat_history.append({"user": user_input, "bot": answer})

        # Display chat
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Insights:** {chat['bot']}")

    # Gene Perturbations
    with tabs[3]:
        st.subheader("Suggested Gene Perturbations")
        st.markdown("These are hypothetical gene perturbations that may reduce lesions:")
        for gene, effects in st.session_state.suggested_genes.items():
            st.markdown(f"- {gene}: immune effect = {effects['immune']}, neuron effect = {effects['neuron']}")
