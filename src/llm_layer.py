"""
LLM Layer for MS Simulation Insights
Handles LLM-based analysis and gene perturbation suggestions
"""

import streamlit as st
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory


def initialize_memory():
    """Initialize conversation memory if not already present."""
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )


def get_groq_api_key():
    """Retrieve Groq API key from Streamlit secrets."""
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in Streamlit secrets")
    return api_key


def format_lesion_data(lesion_matrix, num_samples=-1):
    """Format lesion matrix data for LLM prompt."""
    sample_data = lesion_matrix[:num_samples]
    return "\n".join([
        f"Patient {i+1}: {row.tolist()}"
        for i, row in enumerate(sample_data)
    ])


def create_insights_prompt(lesion_matrix, top_genes, user_question):
    """Create a structured prompt for LLM insights."""
    lesion_summary = format_lesion_data(lesion_matrix)
    
    prompt = f"""You are an expert in Multiple Sclerosis simulations and disease modeling.

LESION DATA (first 5 patients):
{lesion_summary}

GENES OF INTEREST:
{', '.join(top_genes)}

USER QUESTION:
{user_question}

Please provide a clear, informative answer based on the simulation data. Focus on:
- Pattern analysis in lesion progression
- Potential gene-disease relationships
- Clinical implications if applicable

Respond in plain text without markdown formatting."""
    
    return prompt


def generate_insights(lesion_matrix, top_genes=None, user_question=""):
    """
    Generate insights from lesion data using LLM.
    
    Args:
        lesion_matrix: NumPy array of lesion trajectories
        top_genes: List of gene names to consider
        user_question: User's specific question
        
    Returns:
        str: LLM-generated insights or error message
    """
    # Set defaults
    if top_genes is None:
        top_genes = ['GeneA', 'GeneB', 'GeneC']
    
    if not user_question:
        user_question = "What patterns do you observe in the lesion progression?"
    
    # Initialize memory
    initialize_memory()
    
    try:
        # Get API key
        api_key = get_groq_api_key()
        
        # Create prompt
        prompt_text = create_insights_prompt(lesion_matrix, top_genes, user_question)
        
        # Setup LLM chain
        llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
        prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template="{user_input}"
        )
        chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            memory=st.session_state.chat_memory
        )
        
        # Generate response
        result = chain.invoke({"user_input": prompt_text})
        
        # Extract text from result
        if isinstance(result, dict):
            return result.get("text", str(result))
        return str(result)
        
    except ValueError as ve:
        return f"Configuration error: {str(ve)}"
    except Exception as e:
        return f"Error generating insights: {str(e)}"


def generate_gene_effect(min_effect=-0.2, max_effect=0.2, decimals=3):
    """Generate a random gene effect value."""
    return float(np.round(np.random.uniform(min_effect, max_effect), decimals))


def suggest_genes_for_perturbation(lesion_matrix, gene_list):
    """
    Suggest hypothetical gene perturbations based on simulation data.
    
    Args:
        lesion_matrix: NumPy array of lesion trajectories (currently used for validation)
        gene_list: List of gene names to analyze
        
    Returns:
        dict: Mapping of gene names to immune and neuron effects
    """
    if not gene_list:
        return {}
    
    gene_effects = {}
    for gene in gene_list:
        gene_effects[gene] = {
            "immune": generate_gene_effect(),
            "neuron": generate_gene_effect()
        }
    
    return gene_effects