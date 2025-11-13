import streamlit as st
import numpy as np
from src.simulation import simulate_cohort
from src.utils import plot_lesion_matrix, plot_kaplan_meier
from src.llm_layer import generate_insights, suggest_genes_for_perturbation

# Page config
st.set_page_config(page_title="Synthovion", layout="wide")
st.title("🧬 Synthovion: MS Lesion Simulation & Insights")

# Sidebar
st.sidebar.header("Simulation Parameters")
num_patients = st.sidebar.slider("Number of Patients", 5, 50, 20)
observation_days = st.sidebar.slider("Observation Days", 10, 50, 20)
drug_effectiveness = st.sidebar.slider("Drug Effectiveness", 0.0, 1.0, 0.5)
gene_input = st.sidebar.text_input("Hypothetical Genes (comma-separated)", "GeneA,GeneB,GeneC")
gene_list = [g.strip() for g in gene_input.split(",") if g.strip()]

# Initialize session states
if "lesion_matrix" not in st.session_state:
    st.session_state.lesion_matrix = None
if "days" not in st.session_state:
    st.session_state.days = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggested_genes" not in st.session_state:
    st.session_state.suggested_genes = None

# Run simulation button
if st.button("Run Simulation"):
    # Random gene effects
    gene_factors = {
        g: {"immune": float(np.round(np.random.uniform(-0.2,0.2),3)),
            "neuron": float(np.round(np.random.uniform(-0.2,0.2),3))}
        for g in gene_list
    }

    # Simulate cohort
    st.session_state.days, st.session_state.lesion_matrix = simulate_cohort(
        num_patients=num_patients,
        observation_days=observation_days,
        drug_effectiveness=drug_effectiveness,
        gene_factors=gene_factors
    )

    # Suggest gene perturbations
    st.session_state.suggested_genes = suggest_genes_for_perturbation(
        st.session_state.lesion_matrix,
        top_genes=gene_list
    )

# Only show tabs if simulation has run
if st.session_state.lesion_matrix is not None:
    tabs = st.tabs(["Lesion Trajectories", "Kaplan-Meier", "Gene Perturbation", "Chat Insights"])

    # --- Lesion Trajectories ---
    with tabs[0]:
        st.subheader("MS Lesion Trajectories")
        st.pyplot(plot_lesion_matrix(st.session_state.days, st.session_state.lesion_matrix))

    # --- Kaplan-Meier ---
    with tabs[1]:
        st.subheader("Lesion-Free Survival (Synthetic Kaplan-Meier)")
        st.pyplot(plot_kaplan_meier(st.session_state.lesion_matrix, threshold=2.0))

    # --- Gene Perturbation ---
    with tabs[2]:
        st.subheader("Suggested Gene Perturbations")
        st.markdown("These are hypothetical gene perturbations that may reduce lesions:")
        if st.session_state.suggested_genes:
            for gene, effects in st.session_state.suggested_genes.items():
                st.markdown(f"**{gene}**: immune effect = {effects['immune']}, neuron effect = {effects['neuron']}")

    # --- Chat Insights ---
    with tabs[3]:
        st.subheader("Simulation Insights Chat")

        user_input = st.text_input("Ask a question:", key="chat_input")
        if st.button("Send Question", key="chat_btn"):
            if user_input.strip():
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
            st.markdown("---")
