# app.py
import streamlit as st
import numpy as np
from src.simulation import simulate_cohort
from src.utils import plot_lesion_matrix, plot_kaplan_meier
from src.llm_layer import generate_insights, suggest_genes_for_perturbation

st.set_page_config(page_title="Synthovion", layout="wide")
st.title("🧬 Synthovion: MS Lesion Simulation & Insights")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
num_patients = st.sidebar.slider("Number of Patients", 5, 50, 20)
observation_days = st.sidebar.slider("Observation Days", 10, 50, 20)
drug_effectiveness = st.sidebar.slider("Drug Effectiveness", 0.0, 1.0, 0.5)
gene_input = st.sidebar.text_input("Hypothetical Genes (comma-separated)", "GeneA,GeneB,GeneC")
gene_list = [g.strip() for g in gene_input.split(",") if g.strip()]

# --- Run simulation ---
if st.button("Run Simulation"):
    # Generate gene effects for simulation
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

    # Suggest genes for perturbation (only from user input genes)
    suggested_genes = suggest_genes_for_perturbation(lesion_matrix, gene_list)

    # --- Tabs ---
    tabs = st.tabs(["Lesion Trajectories", "Kaplan-Meier", "Chat Insights", "Gene Perturbations"])

    # --- Lesion Trajectories ---
    with tabs[0]:
        st.subheader("MS Lesion Trajectories")
        st.pyplot(plot_lesion_matrix(days, lesion_matrix, max_patients=min(10, num_patients)))

    # --- Kaplan-Meier ---
    with tabs[1]:
        st.subheader("Lesion-Free Survival")
        st.pyplot(plot_kaplan_meier(lesion_matrix, threshold=1.0))  # Use lower threshold for realistic events

    # --- Chat Insights ---
    with tabs[2]:
        st.subheader("Simulation Insights Chat")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question about the simulation:")
            submitted = st.form_submit_button("Send")
            if submitted and user_question:
                answer = generate_insights(lesion_matrix, top_genes=gene_list, user_question=user_question)
                st.session_state.chat_history.append({"user": user_question, "bot": answer})

        # Display chat in order
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Insights:** {chat['bot']}")

    # --- Gene Perturbations ---
    with tabs[3]:
        st.subheader("Suggested Gene Perturbations")
        st.markdown("Hypothetical perturbations to reduce lesions:")
        for gene, effects in suggested_genes.items():
            st.markdown(f"- **{gene}**: immune effect = {effects['immune']}, neuron effect = {effects['neuron']}")
