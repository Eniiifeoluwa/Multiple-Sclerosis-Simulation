# app.py
import streamlit as st
import numpy as np
from src.simulation import simulate_cohort
from src.utils import plot_lesion_matrix, plot_kaplan_meier
from src.llm_layer import generate_insights, suggest_genes_for_perturbation

# Page config
st.set_page_config(page_title="Synthovion", layout="wide")
st.title("🧬 Synthovion: MS Lesion Simulation & Insights")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
num_patients = st.sidebar.slider("Number of Patients", 5, 50, 20)
observation_days = st.sidebar.slider("Observation Days", 10, 50, 20)
drug_effectiveness = st.sidebar.slider("Drug Effectiveness", 0.0, 1.0, 0.5)
gene_input = st.sidebar.text_input("Hypothetical Genes (comma-separated)", "GeneA,GeneB,GeneC")
gene_list = [g.strip() for g in gene_input.split(",") if g.strip()]

# Run simulation button
if st.button("Run Simulation"):
    # Random gene effects for simulation
    gene_factors = {
        g: {"immune": np.random.uniform(-0.2, 0.2), "neuron": np.random.uniform(0.8, 1.2)}
        for g in gene_list
    }

    # Simulate cohort
    days, lesion_matrix = simulate_cohort(
        num_patients=num_patients,
        observation_days=observation_days,
        drug_effectiveness=drug_effectiveness,
        gene_factors=gene_factors
    )

    # Suggest gene perturbations
    suggested_genes = suggest_genes_for_perturbation(lesion_matrix, num_genes=3)

    # Tabs for visualization and chat
    tabs = st.tabs(["Lesion Trajectories", "Kaplan-Meier", "Chat Insights", "Gene Perturbations"])

    # --- Lesion Trajectories ---
    with tabs[0]:
        st.subheader("MS Lesion Trajectories (first 10 patients)")
        st.pyplot(plot_lesion_matrix(days, lesion_matrix, max_patients=min(10, num_patients)))

    # --- Kaplan-Meier ---
    with tabs[1]:
        st.subheader("Lesion-Free Survival (Synthetic Kaplan-Meier)")
        st.pyplot(plot_kaplan_meier(lesion_matrix))

    # --- Chat Insights ---
    with tabs[2]:
        st.subheader("Simulation Insights Chat")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_question = st.text_input("Ask a question about the simulation:", key="user_input")
        if st.button("Send Question", key="send_question"):
            if user_question:
                answer = generate_insights(lesion_matrix, top_genes=gene_list, user_question=user_question)
                st.session_state.chat_history.append({"user": user_question, "bot": answer})

        # Display chat history in order
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Insights:** {chat['bot']}")

    # --- Suggested Gene Perturbations ---
    with tabs[3]:
        st.subheader("Suggested Gene Perturbations")
        st.markdown("These are hypothetical gene perturbations that may reduce lesions:")
        for gene, effects in suggested_genes.items():
            st.markdown(f"- **{gene}**: immune effect = {effects['immune']}, neuron effect = {effects['neuron']}")
