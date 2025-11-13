from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory
import streamlit as st

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")  # use Streamlit secrets

# Persistent memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def generate_insights(lesion_matrix, top_genes=None):
    top_genes = top_genes or ["GeneA", "GeneB", "GeneC"]
    prompt_text = f"""
Hey expert! Here are simulated MS lesion trajectories (first 5 patients):
{lesion_matrix[:5].tolist()}

Hypothetical genes affecting the system: {top_genes}

Explain:
1. Mechanisms behind lesion patterns
2. Potential therapeutic targets
"""
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    prompt = PromptTemplate(input_variables=["user_input"], template="{user_input}")
    chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.chat_memory)
    
    return chain.invoke({"user_input": prompt_text})

def suggest_genes_for_perturbation(lesion_matrix, num_genes=3):
    prompt_text = f"""
Suggest {num_genes} genes to perturb to reduce lesions.
Provide immune_activity and neuron_resilience effects (-0.2 to 0.2)
as a JSON dictionary.
Lesion trajectories (first 5 patients):
{lesion_matrix[:5].tolist()}
"""
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    prompt = PromptTemplate(input_variables=["user_input"], template="{user_input}")
    chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.chat_memory)
    
    response = chain.invoke({"user_input": prompt_text})
    
    import json
    try:
        return json.loads(response)
    except:
        return {"error": "Could not parse LLM output"}
