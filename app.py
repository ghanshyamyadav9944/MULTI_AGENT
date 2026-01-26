


import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_community.memory import ConversationBufferMemory, CombinedMemory, VectorStoreRetrieverMemory
from dotenv import load_dotenv

load_dotenv()

# --- LLM & Tools Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = FAISS.from_texts(["Initial memory"], embedding=embedding_model)
shared_memory = VectorStoreRetrieverMemory(retriever=vector_store.as_retriever())

tools = [
    Tool(name="DuckDuckGoSearch", func=DuckDuckGoSearchRun().run, description="Search web"),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1)),
    ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1))
]

# --- Prompts ---
base_template = """You are a {role}. 
Tools: {tools} | Tool names: {tool_names}
History: {chat_history}
Input: {input}
{agent_scratchpad}"""

RESEARCH_PROMPT = PromptTemplate(input_variables=["input", "agent_scratchpad", "tool_names", "tools", "chat_history"], 
                                template=base_template.format(role="Research Agent", tools="{tools}", tool_names="{tool_names}", chat_history="{chat_history}", input="{input}", agent_scratchpad="{agent_scratchpad}"))

CRITIC_PROMPT = PromptTemplate(input_variables=["input", "agent_scratchpad", "tool_names", "tools", "chat_history"],
                              template="You are a Critic Agent. Evaluate this research for gaps or errors: {input}\n{agent_scratchpad}")

SUMMARY_PROMPT = PromptTemplate(input_variables=["input", "agent_scratchpad", "tool_names", "tools", "chat_history"],
                               template="You are a Summarizer. Condense this: {input}\n{agent_scratchpad}")

EMAIL_PROMPT = PromptTemplate(input_variables=["input", "agent_scratchpad", "tool_names", "tools", "chat_history"],
                             template="You are an Email Agent. Write an email based on: {input}\n{agent_scratchpad}")

def build_agent(prompt):
    memory = CombinedMemory(memories=[ConversationBufferMemory(memory_key="chat_history", return_messages=True), shared_memory])
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

research_agent = build_agent(RESEARCH_PROMPT)
critic_agent = build_agent(CRITIC_PROMPT)
summary_agent = build_agent(SUMMARY_PROMPT)
email_agent = build_agent(EMAIL_PROMPT)



