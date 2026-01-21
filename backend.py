

import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = FastAPI(title="AI Multi-Agent API")

class QueryRequest(BaseModel):
    query: str

# ---------- LLM Configuration ----------
# FIXED: Changed from non-existent 2.5 to 1.5-flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/run")
def run_agents(data: QueryRequest):
    research = llm.invoke(f"Research this topic briefly:\n{data.query}")
    summary = llm.invoke(f"Summarize this in 100–150 words:\n{research.content}")
    email = llm.invoke(f"Write a professional email based on this summary:\n{summary.content}")

    final_output = f"""
### 🔍 Research Findings
{research.content}

---

### 📝 Summary
{summary.content}

---

### 📧 Professional Email
{email.content}
    """
    return {"result": final_output, "status": "success"}

if __name__ == "__main__":
    # Internal cloud networking works best on 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=8080)