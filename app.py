import streamlit as st
from src.simulation import simulate_cohort
from src.llm_layer import generate_insights, suggest_genes_for_perturbation, memory
from src.utils import plot_lesion_matrix, plot_kaplan_meier
import json

st.set_page_config(page_title="MS Simulation + ChatGroq Insights", layout="wide")
st.title("MS Simulation + ChatGroq with Memory")

# --- Inputs ---
num_patients = st.slider("Number of patients", 10, 100, 50, 5)
timesteps = st.slider("Timesteps", 10, 100, 50, 5)
drug_effectiveness = st.slider("Drug effectiveness", 0.0, 1.0, 0.2, 0.05)
simulate_genes = st.checkbox("Initial gene perturbations", value=False)

gene_factors = None
if simulate_genes:
    gene_factors = {
        "GeneA": {"immune": 0.1, "neuron": -0.05},
        "GeneB": {"immune": -0.05, "neuron": 0.1},
    }

grok_api_key = st.text_input("Enter your ChatGroq API Key:", type="password")

# --- Run Simulation ---
if st.button("Run Simulation + ChatGroq"):
    times, lesion_matrix = simulate_cohort(
        num_patients=num_patients,
        timesteps=timesteps,
        drug_effectiveness=drug_effectiveness,
        gene_factors=gene_factors
    )

    st.subheader("Initial Lesion Trajectories")
    plot_lesion_matrix(times, lesion_matrix, max_patients=min(10, num_patients))
    plot_kaplan_meier(lesion_matrix, threshold=5)

    st.subheader("ChatGroq Insights")
    if grok_api_key:
        with st.spinner("Generating insights via ChatGroq…"):
            insights = generate_insights(lesion_matrix, top_genes=list(gene_factors.keys()) if gene_factors else None, api_key=grok_api_key)
        st.text_area("LLM Output", insights, height=300)

        st.subheader("ChatGroq Suggested Gene Perturbations")
        with st.spinner("Asking ChatGroq for new gene perturbations…"):
            suggested_genes = suggest_genes_for_perturbation(lesion_matrix, num_genes=3, api_key=grok_api_key)
        st.json(suggested_genes)

        if suggested_genes:
            st.subheader("Simulation with LLM-Suggested Gene Perturbations")
            times2, lesion_matrix2 = simulate_cohort(
                num_patients=num_patients,
                timesteps=timesteps,
                drug_effectiveness=drug_effectiveness,
                gene_factors=suggested_genes
            )
            plot_lesion_matrix(times2, lesion_matrix2, max_patients=min(10, num_patients))
            plot_kaplan_meier(lesion_matrix2, threshold=5)

    # Save memory (optional)
    if st.checkbox("Save conversation memory"):
        with open("chat_memory.json", "w") as f:
            json.dump(memory.chat_memory.messages, f)
        st.success("Memory saved to chat_memory.json")

    # Load memory (optional)
    if st.checkbox("Load previous conversation memory"):
        try:
            with open("chat_memory.json", "r") as f:
                memory.chat_memory.messages = json.load(f)
            st.success("Memory loaded!")
        except:
            st.error("Failed to load memory")
