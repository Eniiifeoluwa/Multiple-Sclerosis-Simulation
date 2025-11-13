from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory  # persistent conversation memory
import sys, os
sys.path.append(os.path.dirname(__file__))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Global memory object (can be saved/loaded for persistence)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def generate_insights(lesion_matrix, top_genes=None, api_key=None):
    """
    Send simulated data to ChatGroq and return mechanistic insights.
    Remembers past interactions using LangChain ConversationBufferMemory.
    """
    top_genes = top_genes or ["GeneA", "GeneB", "GeneC"]

    prompt_text = f"""
Hey ChatGroq! You are a computational biology expert.
Here are simulated MS lesion trajectories for the first 5 patients:
{lesion_matrix[:5].tolist()}

Hypothetical genes affecting the system: {top_genes}

Please explain in a clear, human-readable way:
1. Mechanisms behind lesion patterns.
2. Potential therapeutic targets or interventions.
"""
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    prompt = PromptTemplate(input_variables=[], template=prompt_text)
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return chain.run({})


def suggest_genes_for_perturbation(lesion_matrix, num_genes=3, api_key=None):
    """
    Ask ChatGroq to propose new gene perturbations based on simulated lesions.
    Uses conversation memory to remember past suggestions.
    Returns a dict: gene -> {"immune": effect, "neuron": effect}
    """
    prompt_text = f"""
Hey ChatGroq! You are a computational biology expert.
Here are MS lesion trajectories for the first 5 patients:
{lesion_matrix[:5].tolist()}

Please suggest {num_genes} hypothetical genes to perturb in order to reduce lesions.
Provide values for immune_activity and neuron_resilience effects (between -0.2 and 0.2).
Return as a JSON dict like:
{{"GeneX": {{"immune": 0.1, "neuron": -0.05}}, ...}}
"""
    llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
    prompt = PromptTemplate(input_variables=[], template=prompt_text)
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    response = chain.run({})
    
    try:
        import json
        gene_dict = json.loads(response)
    except:
        gene_dict = None  # fallback if parsing fails

    return gene_dict
