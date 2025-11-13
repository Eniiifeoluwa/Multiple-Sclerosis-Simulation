# app.py
import streamlit as st
import numpy as np
from src.simulation import simulate_cohort
from src.utils import plot_lesion_matrix, plot_kaplan_meier
from src.llm_layer import generate_insights

st.set_page_config(page_title="Synthovion", layout="wide")
st.title("🧬 Synthovion: MS Lesion Simulation & Insights")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
num_patients = st.sidebar.slider("Number of Patients", 5, 50, 20)
observation_days = st.sidebar.slider("Observation Days", 10, 50, 20)
drug_effectiveness = st.sidebar.slider("Drug Effectiveness", 0.0, 1.0, 0.5)
gene_input = st.sidebar.text_input("Hypothetical Genes (comma-separated)", "GeneA,GeneB,GeneC")
gene_list = [g.strip() for g in gene_input.split(",") if g.strip()]

# Initialize session state
if "lesion_matrix" not in st.session_state:
    st.session_state.lesion_matrix = None
if "days" not in st.session_state:
    st.session_state.days = None
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

    # Store in session state
    st.session_state.lesion_matrix = lesion_matrix
    st.session_state.days = days
    st.success("Simulation complete!")

# --- Tabs ---
tabs = st.tabs(["Lesion Trajectories", "Kaplan-Meier", "Chat Insights"])

# Trajectories Tab
with tabs[0]:
    st.subheader("MS Lesion Trajectories")
    if st.session_state.lesion_matrix is not None:
        st.pyplot(plot_lesion_matrix(st.session_state.days, st.session_state.lesion_matrix))
    else:
        st.info("Run the simulation first to see lesion trajectories.")

# Kaplan-Meier Tab
with tabs[1]:
    st.subheader("Lesion-Free Survival (Synthetic Kaplan-Meier)")
    if st.session_state.lesion_matrix is not None:
        st.pyplot(plot_kaplan_meier(st.session_state.lesion_matrix))
    else:
        st.info("Run the simulation first to see Kaplan-Meier curve.")

# Chat Insights Tab
with tabs[2]:
    st.subheader("Simulation Insights Chat")
    if st.session_state.lesion_matrix is None:
        st.info("Run the simulation first to enable chat insights.")
    else:
        with st.form("chat_form", clear_on_submit=False):
            user_input = st.text_input("Ask a question:")
            submitted = st.form_submit_button("Send")
            if submitted and user_input:
                try:
                    answer = generate_insights(st.session_state.lesion_matrix, top_genes=gene_list)
                except Exception as e:
                    answer = f"Error generating insights: {str(e)}"
                st.session_state.chat_history.append({"user": user_input, "bot": answer})

        # Display chat history
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Insights:** {chat['bot']}")
