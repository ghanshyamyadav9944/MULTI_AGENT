
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="AI Multi-Agent System", layout="wide", page_icon="🤖")
st.title("🤖 AI Multi-Agent Orchestrator")

api_keys = st.secrets.get("GOOGLE_API_KEYS", [os.getenv("GOOGLE_API_KEY")])

def get_agent_response(agent_role, prompt_text):
    """Helper to simulate agent work with key rotation"""
    for key in api_keys:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=key)
            return llm.invoke(prompt_text).content
        except Exception as e:
            if "429" in str(e): continue
            st.error(f"Error: {e}")
            return None
    return None

query = st.text_area("Enter your research topic:", placeholder="e.g. Next-gen battery technologies")

if st.button("Start Multi-Agent Workflow"):
    if query:
        # 1. Create Tabs first (they stay empty until agents finish)
        t_res, t_crit, t_sum, t_em = st.tabs(["🔍 Research", "⚖️ Critic Review", "📝 Summary", "📧 Email Draft"])
        
        # 2. Status container for real-time tracking
        with st.status("Agent Pipeline Initiated...", expanded=True) as status:
            
            st.write("📡 **Research Agent** is searching the web...")
            res_out = get_agent_response("Research", f"Detailed research on: {query}")
            t_res.markdown(res_out)
            
            st.write("⚖️ **Critic Agent** is analyzing data integrity...")
            crit_out = get_agent_response("Critic", f"Critique this research and find gaps: {res_out}")
            t_crit.markdown(crit_out)
            
            st.write("📝 **Summarizer Agent** is synthesizing findings...")
            sum_out = get_agent_response("Summarizer", f"Summarize this research: {res_out} while considering this critique: {crit_out}")
            t_sum.markdown(sum_out)
            
            st.write("📧 **Email Agent** is drafting the communication...")
            em_out = get_agent_response("Email", f"Write a professional email based on: {sum_out}")
            t_em.markdown(em_out)
            
            status.update(label="✅ Workflow Complete!", state="complete", expanded=False)
        
        st.success("All agents have finished their tasks. Check the tabs above!")
    else:
        st.warning("Please enter a query first.")
