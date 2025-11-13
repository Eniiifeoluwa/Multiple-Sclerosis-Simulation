"""
Synthovion: MS Lesion Simulation & Insights Platform
A Streamlit app for simulating Multiple Sclerosis progression with LLM-powered insights
"""

import streamlit as st
import numpy as np
from src.simulation import simulate_cohort
from src.utils import plot_lesion_matrix, plot_kaplan_meier
from src.llm_layer import generate_insights, suggest_genes_for_perturbation


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Synthovion - MS Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 Synthovion: MS Lesion Simulation & Insights")
st.markdown("""
Explore Multiple Sclerosis lesion progression through simulation and AI-powered analysis.
Adjust parameters in the sidebar and run simulations to see results.
""")


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.header("⚙️ Simulation Parameters")

num_patients = st.sidebar.slider(
    "Number of Patients",
    min_value=5,
    max_value=50,
    value=20,
    help="Total number of patients in the simulated cohort"
)

observation_days = st.sidebar.slider(
    "Observation Days",
    min_value=10,
    max_value=50,
    value=20,
    help="Number of days to track each patient"
)

drug_effectiveness = st.sidebar.slider(
    "Drug Effectiveness",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Treatment effectiveness (0 = no effect, 1 = complete suppression)"
)

gene_input = st.sidebar.text_input(
    "Hypothetical Genes (comma-separated)",
    value="GeneA,GeneB,GeneC",
    help="Enter gene names to include in the analysis"
)

# Parse gene list
gene_list = [gene.strip() for gene in gene_input.split(",") if gene.strip()]

if not gene_list:
    st.sidebar.warning("⚠️ Please enter at least one gene name")


# ============================================================================
# SIMULATION EXECUTION
# ============================================================================

st.sidebar.markdown("---")

if st.sidebar.button("▶️ Run Simulation", type="primary", use_container_width=True):
    with st.spinner("Running simulation..."):
        try:
            # Generate random gene effects
            gene_factors = {
                gene: {
                    "immune": np.random.uniform(-0.2, 0.2),
                    "neuron": np.random.uniform(-0.2, 0.2)
                }
                for gene in gene_list
            }
            
            # Run cohort simulation
            days, lesion_matrix = simulate_cohort(
                num_patients=num_patients,
                observation_days=observation_days,
                drug_effectiveness=drug_effectiveness,
                gene_factors=gene_factors
            )
            
            # Store results in session state
            st.session_state.lesion_matrix = lesion_matrix
            st.session_state.days = days
            st.session_state.gene_factors = gene_factors
            st.session_state.gene_list = gene_list
            
            # Generate gene perturbation suggestions
            st.session_state.suggested_genes = suggest_genes_for_perturbation(
                lesion_matrix,
                gene_list
            )
            
            # Initialize chat history if needed
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            st.sidebar.success("✅ Simulation complete!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Simulation failed: {str(e)}")


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

if "lesion_matrix" not in st.session_state:
    st.info("👈 Configure parameters in the sidebar and click 'Run Simulation' to begin")
    st.stop()


# Create tabs for different views
tab_trajectories, tab_kaplan, tab_chat, tab_genes = st.tabs([
    "📈 Lesion Trajectories",
    "📊 Kaplan-Meier Analysis",
    "💬 Chat Insights",
    "🧬 Gene Perturbations"
])


# ----------------------------------------------------------------------------
# TAB 1: Lesion Trajectories
# ----------------------------------------------------------------------------

with tab_trajectories:
    st.subheader("MS Lesion Trajectories")
    st.markdown("""
    Each line represents a simulated patient's lesion count over time.
    Trajectories vary based on individual immune factors, neuron resilience, and gene effects.
    """)
    
    fig = plot_lesion_matrix(
        st.session_state.days,
        st.session_state.lesion_matrix,
        max_patients=10
    )
    st.pyplot(fig)
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Average Final Lesion Count",
            f"{st.session_state.lesion_matrix[:, -1].mean():.2f}"
        )
    with col2:
        st.metric(
            "Max Lesion Count Observed",
            f"{st.session_state.lesion_matrix.max():.2f}"
        )
    with col3:
        st.metric(
            "Patients with Progressive Disease",
            f"{(st.session_state.lesion_matrix[:, -1] > st.session_state.lesion_matrix[:, 0]).sum()}"
        )


# ----------------------------------------------------------------------------
# TAB 2: Kaplan-Meier Analysis
# ----------------------------------------------------------------------------

with tab_kaplan:
    st.subheader("Lesion-Free Survival Analysis")
    st.markdown("""
    This Kaplan-Meier curve shows the probability of remaining below the lesion threshold over time.
    An "event" occurs when a patient's lesion count crosses the specified threshold.
    """)
    
    # Threshold selector
    threshold = st.slider(
        "Lesion Count Threshold",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Define what lesion count constitutes a significant event"
    )
    
    fig = plot_kaplan_meier(st.session_state.lesion_matrix, threshold=threshold)
    st.pyplot(fig)


# ----------------------------------------------------------------------------
# TAB 3: Chat Insights
# ----------------------------------------------------------------------------

with tab_chat:
    st.subheader("🤖 AI-Powered Simulation Insights")
    st.markdown("""
    Ask questions about the simulation results and receive AI-generated insights
    based on the lesion data and gene factors.
    """)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # User input
    user_input = st.text_input(
        "Your question:",
        placeholder="e.g., What patterns do you see in the lesion progression?",
        key="chat_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Send", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Process question
    if send_button and user_input:
        with st.spinner("Generating insights..."):
            answer = generate_insights(
                st.session_state.lesion_matrix,
                top_genes=st.session_state.gene_list,
                user_question=user_input
            )
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": answer
            })
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**👤 You:** {chat['user']}")
                st.markdown(f"**🤖 AI Insights:** {chat['bot']}")
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")
    else:
        st.info("💡 No questions yet. Ask something about the simulation!")


# ----------------------------------------------------------------------------
# TAB 4: Gene Perturbations
# ----------------------------------------------------------------------------

with tab_genes:
    st.subheader("🧬 Suggested Gene Perturbations")
    st.markdown("""
    These are hypothetical gene perturbations that may influence lesion dynamics.
    **Immune effects** modify inflammatory attack intensity, while **neuron effects**
    impact recovery and resilience.
    """)
    
    if st.session_state.get("suggested_genes"):
        # Create a formatted table
        gene_data = []
        for gene, effects in st.session_state.suggested_genes.items():
            gene_data.append({
                "Gene": gene,
                "Immune Effect": f"{effects['immune']:+.3f}",
                "Neuron Effect": f"{effects['neuron']:+.3f}",
                "Overall Impact": "↑ Attack" if effects['immune'] > 0 else "↓ Attack"
            })
        
        import pandas as pd
        df = pd.DataFrame(gene_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Interpretation:**
        - **Positive immune effect**: Increases inflammatory attack → more lesions
        - **Negative immune effect**: Reduces inflammatory attack → fewer lesions
        - **Positive neuron effect**: Enhances recovery → fewer lesions
        - **Negative neuron effect**: Impairs recovery → more lesions
        """)
    else:
        st.warning("⚠️ Run the simulation first to see gene perturbations.")


# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About Synthovion
A simulation platform for exploring MS disease progression
with AI-powered insights and gene perturbation analysis.
""")