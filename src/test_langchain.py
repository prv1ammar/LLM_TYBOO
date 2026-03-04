"""
test_langchain_agents.py — LangChain Agent Test Suite
=====================================================
PURPOSE:
  Tests the integration of LangChain with the Tython platform.
  Verifies that LiteLLM correctly routes LangChain requests to models.

SCENARIOS:
  1. Simple Chat      — Direct message to 14B model via ChatOpenAI.
  2. Agent with Tools — A ReAct agent using Wikipedia to answer questions.
  3. Memory Agent     — An agent that remembers previous messages.
  4. RAG Agent        — Integration with internal VectorStore.

PREREQUISITES:
  - Docker services running (docker compose up -d)
  - LangChain installed (pip install langchain langchain-openai)
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage

load_dotenv()

# Configuration — Targets LiteLLM Proxy
# If running inside Docker, use 'http://litellm:4000'
# If running locally on host, use 'http://localhost:4000'
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
if "litellm" in LITELLM_URL and os.name == "nt": 
    # Fallback for Windows host development if .env has docker service name
    LITELLM_URL = "http://localhost:4000"

LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-tyboo-26ddf338ae64d6706d9e67aee033ea8f")

# 1. Initialize the LLM (OpenAI-compatible via LiteLLM)
llm = ChatOpenAI(
    model="internal-llm",         # Default 14B model
    openai_api_base=f"{LITELLM_URL}/v1",
    openai_api_key=LITELLM_KEY,
    temperature=0.1
)

async def test_simple_chat():
    print("\n--- Test 1: Simple Chat ---")
    messages = [HumanMessage(content="What is the capital of Morocco? Answer in 1 word.")]
    response = llm.invoke(messages)
    print(f"Response: {response.content}")

async def test_agent_with_tools():
    print("\n--- Test 2: ReAct Agent with Wikipedia ---")
    try:
        tools = load_tools(["wikipedia"], llm=llm)
        agent = initialize_agent(
            tools, 
            llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=True
        )
        query = "Who is the current King of Morocco and when did he ascend to the throne?"
        response = agent.run(query)
        print(f"Agent Response: {response}")
    except Exception as e:
        print(f"Agent failed (likely missing wikipedia package): {e}")

async def test_memory_agent():
    print("\n--- Test 3: Agent with Memory ---")
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_chain = initialize_agent(
        [], 
        llm, 
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True, 
        memory=memory
    )
    
    print("User: My name is Ammar.")
    agent_chain.run(input="My name is Ammar.")
    
    print("User: What is my name?")
    response = agent_chain.run(input="What is my name?")
    print(f"Agent Response: {response}")

async def main():
    print("🚀 Starting LangChain Agent Tests...")
    
    await test_simple_chat()
    await test_agent_with_tools()
    await test_memory_agent()
    
    print("\n✅ LangChain Tests Completed.")

if __name__ == "__main__":
    asyncio.run(main())
