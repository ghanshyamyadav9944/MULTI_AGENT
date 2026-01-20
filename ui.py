



# ui.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8080"

st.set_page_config(page_title="AI Multi Agent", layout="centered")
st.title("🤖 AI MULTI AGENT")

query = st.text_area("Enter your query", height=150)

if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Running agents..."):
            try:
                health = requests.get(f"{API_URL}/health", timeout=3)
                if health.status_code != 200:
                    st.error("❌ FastAPI server not running")
                else:
                    response = requests.post(
                        f"{API_URL}/run",
                        json={"query": query},
                        timeout=300
                    )
                    st.success("✅ Completed")
                    st.write(response.json()["result"])
            except Exception as e:
                st.error(f"Connection error: {e}")
