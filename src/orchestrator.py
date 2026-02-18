"""
Advanced multi-agent orchestration using PydanticAI
Demonstrates how to build complex workflows with specialized agents
"""
import os
from typing import List, Dict
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from embeddings import EmbeddingService
from dotenv import load_dotenv

load_dotenv()

LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-1234")

# Initialize the model
from pydantic_ai.providers.openai import OpenAIProvider

# Initialize the model with custom provider
provider = OpenAIProvider(
    base_url=f"{LITELLM_URL}/v1",
    api_key=LITELLM_KEY,
)

model = OpenAIModel(
    'internal-llm',
    provider=provider
)

# Define structured outputs
class AnalysisResult(BaseModel):
    """Structured analysis result"""
    summary: str
    key_points: List[str]
    sentiment: str
    confidence: float

class TaskDecomposition(BaseModel):
    """Break down complex tasks into subtasks"""
    main_task: str
    subtasks: List[str]
    estimated_complexity: str

from production_rag import ProductionRAG

# specialized Agents
class AgentOrchestrator:
    """Orchestrates multiple specialized agents with RAG capabilities"""
    
    def __init__(self):
        self.rag = ProductionRAG(collection_name="knowledge_base")
        
        # General-Purpose Enterprise Agent
        self.general_agent = Agent(
            model,
            system_prompt="""You are AGATE, a general-purpose enterprise AI assistant.
            You can help with any topic: legal questions, HR, IT, finance, strategy, 
            drafting documents, analysis, or any other professional task.
            
            Depending on the user's question, you should:
            1. Use 'search_knowledge_base' if you need to find internal documents, policies, or technical guides.
            2. Answer directly if the question is general knowledge or creative.
            
            Always respond in the same language as the user (Arabic, French, or English).
            Be professional, concise, and helpful."""
        )

        # Register tools for the general agent
        @self.general_agent.tool_plain
        async def search_knowledge_base(query: str) -> str:
            """Search the enterprise knowledge base for relevant documents."""
            result = await self.rag.query(query, top_k=3)
            return f"CONTEXT FROM KB:\n{result['answer']}"

        @self.general_agent.tool_plain
        async def send_email(recipient: str, subject: str, body: str) -> str:
            """Send an email for official communication."""
            print(f"📧 [SIMULATED EMAIL] To: {recipient}, Subject: {subject}")
            return f"Email sent to {recipient}"

        @self.general_agent.tool_plain
        async def post_to_slack(channel: str, message: str) -> str:
            """Post a notification or alert to a Slack channel."""
            print(f"💬 [SIMULATED SLACK] Channel: #{channel}, Message: {message}")
            return f"Posted to Slack channel #{channel}"

    async def run_agent(self, query: str) -> str:
        """Handle any enterprise inquiry using the general agent"""
        result = await self.general_agent.run(query)
        return result.data

# Example usage
async def main():
    print("🤖 AGATE General-Purpose Enterprise Demo\n")
    orchestrator = AgentOrchestrator()
    
    # 1. Legal Query
    print("⚖️  Example 1: Legal Consultation")
    legal_query = "What are the legal requirements for employee overtime in Morocco?"
    legal_response = await orchestrator.run_agent(legal_query)
    print(f"Query: {legal_query}")
    print(f"Agent: {legal_response}\n")
    
    # 2. HR Query
    print("👥 Example 2: HR Support")
    hr_query = "Please email the holiday policy summary to manager@company.ma"
    hr_response = await orchestrator.run_agent(hr_query)
    print(f"Query: {hr_query}")
    print(f"Agent: {hr_response}\n")

    # 3. IT Query
    print("💻 Example 3: IT Troubleshooting")
    it_query = "If the vLLM container is down, please post an alert to the #devops-alerts channel."
    it_response = await orchestrator.run_agent(it_query)
    print(f"Query: {it_query}")
    print(f"Agent: {it_response}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
