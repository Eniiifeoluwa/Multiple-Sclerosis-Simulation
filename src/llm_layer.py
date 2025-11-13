from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory

import os
import json

# Global memory for chat
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_insights(lesion_matrix, top_genes=None):
    """Return mechanistic insights in a chat-like manner."""
    top_genes = top_genes or ["GeneA", "GeneB", "GeneC"]
    prompt_text = f"""
You are a computational biology expert.
Here are simulated lesion trajectories for the first 5 patients:
{lesion_matrix[:5].tolist()}

Hypothetical genes affecting the system: {top_genes}

Please explain in clear, human-readable language:
1. Patterns of lesion progression.
2. Potential therapeutic interventions.
"""
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    prompt = PromptTemplate(input_variables=["user_input"], template="{user_input}")
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return chain.run({"user_input": prompt_text})

def suggest_genes_for_perturbation(lesion_matrix, top_genes=None, num_genes=3):
    """Suggest hypothetical gene perturbations."""
    top_genes = top_genes or ["GeneA", "GeneB", "GeneC"]
    prompt_text = f"""
You are a computational biology expert.
Here are lesion trajectories for the first 5 patients:
{lesion_matrix[:5].tolist()}

Suggest {num_genes} hypothetical genes to perturb to reduce lesions.
Return a JSON dict like: {{"GeneX": {{"immune": 0.1, "neuron": -0.05}}, ...}}
"""
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    prompt = PromptTemplate(input_variables=["user_input"], template="{user_input}")
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    try:
        response = chain.run({"user_input": prompt_text})
        gene_dict = json.loads(response)
    except:
        gene_dict = None
    return gene_dict
