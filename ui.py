

import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Setup Page
st.set_page_config(page_title="AI Multi Agent", layout="centered")
st.title("🤖 AI MULTI AGENT ")

# 2. Get the list of keys from secrets
api_keys = st.secrets.get("GOOGLE_API_KEYS", [])

def get_llm_response(prompt):
    """Tries to get a response using available API keys in order."""
    if not api_keys:
        st.error("No API keys found in secrets!")
        return None

    for i, key in enumerate(api_keys):
        try:
            # Initialize LLM with the current key
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                google_api_key=key
            )
            return llm.invoke(prompt)
        
        except Exception as e:
            # Check if error is related to Quota (429)
            if "429" in str(e) or "quota" in str(e).lower():
                st.warning(f"⚠️ Key {i+1} quota exceeded. Switching to Key {i+2}...")
                continue # Try the next key in the list
            else:
                # If it's a different error (like network), stop and show it
                st.error(f"Error with Key {i+1}: {e}")
                return None
    
    st.error("❌ All API keys have exceeded their daily quota.")
    return None

# 3. UI logic
query = st.text_area("Enter your query", height=150)

if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Agents are working..."):
            # Execute agents using our rotation helper
            research = get_llm_response(f"Research this topic briefly:\n{query}")
            
            if research:
                summary = get_llm_response(f"Summarize this in 100 words:\n{research.content}")
                if summary:
                    email = get_llm_response(f"Write a professional email based on:\n{summary.content}")
                    
                    if email:
                        st.success("✅ Completed successfully using available quota!")
                        st.markdown("### 🔍 Research\n" + research.content)
                        st.markdown("### 📝 Summary\n" + summary.content)
                        st.markdown("### 📧 Email\n" + email.content)