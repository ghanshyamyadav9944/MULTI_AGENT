



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
@app.post("/run")
def run_agents(data: QueryRequest):
    # Agent 1: Research
    research = llm.invoke(f"Research: {data.query}")
    # Agent 2: Critic
    critic = llm.invoke(f"Critique this research: {research.content}")
    # Agent 3: Summary
    summary = llm.invoke(f"Summarize: {research.content} taking into account: {critic.content}")
    # Agent 4: Email
    email = llm.invoke(f"Draft email: {summary.content}")

    return {
        "research": research.content,
        "critic": critic.content,
        "summary": summary.content,
        "email": email.content,
        "status": "success"
    }



