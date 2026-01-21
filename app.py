

import os
import json
from typing import Dict, Any, List
from pydantic import BaseModel
from dotenv import load_dotenv

# --- 1st step: Imports ---
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Updated Imports for Tools
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun

from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_community.memory import (
    ConversationBufferMemory,
    CombinedMemory,
    VectorStoreRetrieverMemory
)

load_dotenv()

# --- 2nd step: Initialize LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Updated to valid model
    temperature=0.2
)

# --- 3rd step: Embeddings & Shared Vector Memory ---
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
vector_store = FAISS.from_texts(
    texts=["Initial memory"],
    embedding=embedding_model
)
shared_memory = VectorStoreRetrieverMemory(
    retriever=vector_store.as_retriever()
)

# --- 4th step: Tools Setup (Including Wiki & ArXiv) ---

# 1. Setup DuckDuckGo
duckduckgo = DuckDuckGoSearchRun()

# 2. Setup Wikipedia
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# 3. Setup ArXiv 
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Combine into the tools list
tools = [
    Tool(
        name="DuckDuckGoSearch",
        func=duckduckgo.run,
        description="Search the web using DuckDuckGo"
    ),
    wiki_tool,
    arxiv_tool
]

# --- 5th step: Agent Prompts ---
# (Prompts remain the same as your original snippet)
RESEARCH_PROMPT = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools", "chat_history"],
    template="""You are a Research Agent. 
You have access to the following tools: {tools}
Tool names: {tool_names}
Chat history: {chat_history}
Research query: {input}
{agent_scratchpad}"""
)

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools", "chat_history"],
    template="""You are a Summarizer Agent.
Summarize the following content: {input}
{agent_scratchpad}"""
)

EMAIL_PROMPT = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools", "chat_history"],
    template="""You are an Email Agent.
Write a professional email based on: {input}
{agent_scratchpad}"""
)

# --- 6th step: Individual Memories ---
def build_memory():
    conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return CombinedMemory(
        memories=[conversation_memory, shared_memory]
    )

# --- 8th step: Create Specialized Agents ---
def build_agent(prompt):
    memory = build_memory()
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

research_agent = build_agent(RESEARCH_PROMPT)
summary_agent = build_agent(SUMMARY_PROMPT)
email_agent = build_agent(EMAIL_PROMPT)

# --- 9th step: Supervisor Orchestration ---
def supervisor_workflow(query: str) -> str:
    research_output = research_agent.invoke({"input": query})["output"]
    summary_output = summary_agent.invoke({"input": research_output})["output"]
    email_output = email_agent.invoke({"input": summary_output})["output"]
    return email_output

if __name__ == "__main__":
    result = supervisor_workflow("Latest trends in AI agent orchestration")
    print(result)