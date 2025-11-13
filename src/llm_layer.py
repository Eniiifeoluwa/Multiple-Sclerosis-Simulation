from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory
import os, json

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Persistent memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def generate_insights(lesion_matrix, top_genes=None, api_key=None):
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

    llm = ChatGroq(api_key=api_key or GROQ_API_KEY, model="llama-3.1-8b-instant")

    # Use an explicit input variable
    prompt = PromptTemplate(input_variables=["user_input"], template="{user_input}")

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # Provide input explicitly
    return chain.invoke({"user_input": prompt_text})


def suggest_genes_for_perturbation(lesion_matrix, num_genes=3, api_key=None):
    prompt_text = f"""
Hey guy! You are a computational biology expert.
Here are MS lesion trajectories for the first 5 patients:
{lesion_matrix[:5].tolist()}

Please suggest {num_genes} hypothetical genes to perturb in order to reduce lesions.
Provide values for immune_activity and neuron_resilience effects (between -0.2 and 0.2).
Return as a JSON dict like:
{{"GeneX": {{"immune": 0.1, "neuron": -0.05}}, ...}}
"""

    llm = ChatGroq(api_key=api_key or GROQ_API_KEY, model="llama-3.1-8b-instant")

    prompt = PromptTemplate(input_variables=["user_input"], template="{user_input}")
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    response = chain.invoke({"user_input": prompt_text})

    try:
        gene_dict = json.loads(response)
    except:
        gene_dict = None

    return gene_dict
