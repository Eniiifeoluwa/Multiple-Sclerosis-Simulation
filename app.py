import streamlit as st
import numpy as np
from src.simulation import simulate_cohort
from src.utils import plot_lesion_matrix, plot_kaplan_meier
from src.llm_layer import generate_insights, suggest_genes_for_perturbation

st.set_page_config(page_title="Synthovion", layout="wide")
st.title("🧬 Synthovion: MS Lesion Simulation & Insights")

# Sidebar
st.sidebar.header("Simulation Parameters")
num_patients = st.sidebar.slider("Number of Patients", 5, 50, 20)
observation_days = st.sidebar.slider("Observation Days", 10, 50, 20)
drug_effectiveness = st.sidebar.slider("Drug Effectiveness", 0.0, 1.0, 0.5)
gene_input = st.sidebar.text_input("Hypothetical Genes (comma-separated)", "GeneA,GeneB,GeneC")
gene_list = [g.strip() for g in gene_input.split(",") if g.strip()]

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Run Simulation ---
if st.button("Run Simulation"):
    gene_factors = {
        g: {"immune": np.random.uniform(-0.2,0.2), "neuron": np.random.uniform(-0.2,0.2)}
        for g in gene_list
    }
    days, lesion_matrix = simulate_cohort(
        num_patients=num_patients,
        observation_days=observation_days,
        drug_effectiveness=drug_effectiveness,
        gene_factors=gene_factors
    )
    st.session_state["lesion_matrix"] = lesion_matrix
    st.session_state["days"] = days

# --- Tabs ---
tabs = st.tabs(["Lesion Trajectories", "Kaplan-Meier", "Simulation Chat"])

# Lesion Trajectories Tab
with tabs[0]:
    st.subheader("MS Lesion Trajectories")
    if "lesion_matrix" in st.session_state:
        st.pyplot(plot_lesion_matrix(st.session_state["days"], st.session_state["lesion_matrix"]))
    else:
        st.info("Run the simulation to see lesion trajectories.")

# Kaplan-Meier Tab
with tabs[1]:
    st.subheader("Lesion-Free Survival")
    if "lesion_matrix" in st.session_state:
        st.pyplot(plot_kaplan_meier(st.session_state["lesion_matrix"]))
    else:
        st.info("Run the simulation to see survival plot.")

# Chat Tab
with tabs[2]:
    st.subheader("Simulation Insights Chat")
    user_input = st.text_input("Ask a question about the simulation:", key="chat_input")

    if st.button("Send Question"):
        if "lesion_matrix" not in st.session_state:
            st.warning("Run the simulation first!")
        elif user_input.strip():
            response = generate_insights(st.session_state["lesion_matrix"], top_genes=gene_list)
            st.session_state.chat_history.append({"user": user_input, "bot": response})

    # Display conversation
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Insights:** {chat['bot']}")
