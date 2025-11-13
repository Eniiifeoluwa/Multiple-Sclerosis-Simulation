# llm_layer.py
import streamlit as st
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory
import json

# Load API key from Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

# Initialize persistent memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

def generate_insights(lesion_matrix, top_genes=None, user_question=""):
    prompt_text = f"""
You are an expert in Multiple Sclerosis simulations.
Here are the lesion trajectories for the first 5 patients:
{lesion_matrix[:5].tolist()}

Hypothetical genes affecting the system: {top_genes or ['GeneA','GeneB','GeneC']}

Answer the following question in plain text:
{user_question}
"""
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    prompt = PromptTemplate(input_variables=["user_input"], template="{user_input}")
    chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.chat_memory)

    result = chain.invoke({"user_input": prompt_text})

    # Always return as plain text
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    return str(result)



def suggest_genes_for_perturbation(lesion_matrix, num_genes=3):
    """
    Suggest hypothetical gene perturbations with random effects.
    Returns a dictionary: gene -> {"immune": effect, "neuron": effect}
    """
    # Pick random genes
    possible_genes = [f"Gene{chr(i)}" for i in range(65, 91)]  # GeneA-GeneZ
    chosen_genes = np.random.choice(possible_genes, num_genes, replace=False)
    
    gene_dict = {}
    for gene in chosen_genes:
        gene_dict[gene] = {
            "immune": float(np.round(np.random.uniform(-0.2, 0.2), 3)),
            "neuron": float(np.round(np.random.uniform(-0.2, 0.2), 3))
        }
    
    # Add explanation prompt
    prompt_text = f"""
You are an MS simulation expert.
Here are lesion trajectories for the first 5 patients:
{lesion_matrix[:5].tolist()}

Suggest {num_genes} hypothetical gene perturbations to reduce lesions.
Explain in simple terms what each gene might do.
Return a JSON dictionary like:
{json.dumps(gene_dict, indent=2)}
"""
    # Setup LLM
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    prompt = PromptTemplate(input_variables=["user_input"], template="{user_input}")
    chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.chat_memory)
    
    try:
        response = chain.invoke({"user_input": prompt_text})
        # Ensure JSON output
        gene_dict_llm = json.loads(response)
    except:
        gene_dict_llm = gene_dict  # fallback to random values
    
    return gene_dict_llm
