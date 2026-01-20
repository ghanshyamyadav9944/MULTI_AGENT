
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="AI Multi-Agent API")

# ---------- Models ----------
class QueryRequest(BaseModel):
    query: str

# ---------- LLM Configuration ----------
# Changed model to "gemini-2.5-flash" as "1.5" is not a valid version.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "AI Multi-Agent API is running", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/run")
def run_agents(data: QueryRequest):
    # Agent 1: Research
    research = llm.invoke(f"Research this topic briefly:\n{data.query}")
    
    # Agent 2: Summary
    summary = llm.invoke(
        f"Summarize this in 100–150 words:\n{research.content}"
    )
    
    # Agent 3: Email
    email = llm.invoke(
        f"Write a professional email based on this summary:\n{summary.content}"
    )

    # CRITICAL FIX: Combine all outputs into a single 'result' key 
    # so that ui.py (response.json()["result"]) can read it correctly.
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

    return {
        "result": final_output,
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)