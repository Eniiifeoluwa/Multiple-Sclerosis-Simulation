from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory
import os

# Conversation memory persists across chat interactions
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_insights(lesion_matrix, top_genes=None):
    top_genes = top_genes or ["GeneA","GeneB","GeneC"]
    prompt_text = f"""
You are a computational biology expert.
Here are simulated lesion trajectories (first 5 patients):
{lesion_matrix[:5].tolist()}

Hypothetical genes affecting the system: {top_genes}

Explain:
1. Lesion trajectory patterns in layman-friendly terms.
2. Potential therapeutic targets or interventions.
"""
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    prompt = PromptTemplate(input_variables=[], template=prompt_text)
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return chain.run({})

def suggest_genes_for_perturbation(lesion_matrix, num_genes=3):
    prompt_text = f"""
You are a computational biology expert.
Here are simulated lesion trajectories (first 5 patients):
{lesion_matrix[:5].tolist()}

Suggest {num_genes} genes to perturb to reduce lesions.
Return JSON: {{ "GeneX": {{"immune": 0.1, "neuron": -0.05}}, ... }}
"""
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    prompt = PromptTemplate(input_variables=[], template=prompt_text)
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    import json
    try:
        response = chain.run({})
        return json.loads(response)
    except:
        return None
