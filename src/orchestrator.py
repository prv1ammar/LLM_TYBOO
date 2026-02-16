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

# Specialized Agents
class AgentOrchestrator:
    """Orchestrates multiple specialized agents for complex workflows"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        
        # Analyst Agent - for document analysis
        self.analyst = Agent(
            model,
            output_type=AnalysisResult,
            system_prompt="""You are a professional business analyst for Moroccan enterprises.
            Analyze documents and provide structured insights.
            Be concise, accurate, and highlight key business implications."""
        )
        
        # Planner Agent - for task decomposition
        self.planner = Agent(
            model,
            output_type=TaskDecomposition,
            system_prompt="""You are a strategic planner.
            Break down complex business tasks into actionable subtasks.
            Consider resource constraints and prioritize effectively."""
        )
        
        # Writer Agent - for content generation
        self.writer = Agent(
            model,
            output_type=str,
            system_prompt="""You are a professional business writer.
            Create clear, professional content for Moroccan business contexts.
            Use formal but accessible language."""
        )
    
    async def analyze_document(self, document: str) -> AnalysisResult:
        """Analyze a business document"""
        result = await self.analyst.run(f"Analyze this document:\n\n{document}")
        return result.data
    
    async def plan_task(self, task_description: str) -> TaskDecomposition:
        """Break down a complex task"""
        result = await self.planner.run(f"Create a plan for: {task_description}")
        return result.data
    
    async def generate_content(self, prompt: str, context: str = "") -> str:
        """Generate business content"""
        full_prompt = f"{prompt}\n\nContext: {context}" if context else prompt
        result = await self.writer.run(full_prompt)
        return result.data
    
    async def workflow_example(self, business_goal: str) -> Dict:
        """
        Example of a multi-step workflow:
        1. Plan the task
        2. Generate content for each subtask
        3. Analyze the results
        """
        print(f"🎯 Starting workflow for: {business_goal}\n")
        
        # Step 1: Plan
        print("📋 Step 1: Planning...")
        plan = await self.plan_task(business_goal)
        print(f"   Main task: {plan.main_task}")
        print(f"   Subtasks: {len(plan.subtasks)}")
        print(f"   Complexity: {plan.estimated_complexity}\n")
        
        # Step 2: Execute subtasks
        print("⚙️  Step 2: Executing subtasks...")
        results = []
        for i, subtask in enumerate(plan.subtasks[:3], 1):  # Limit to 3 for demo
            print(f"   {i}. {subtask}")
            content = await self.generate_content(
                f"Provide a brief solution for: {subtask}",
                context=business_goal
            )
            results.append({"subtask": subtask, "solution": content})
        print()
        
        # Step 3: Analyze overall results
        print("📊 Step 3: Analyzing results...")
        combined_results = "\n\n".join([
            f"Subtask: {r['subtask']}\nSolution: {r['solution']}"
            for r in results
        ])
        analysis = await self.analyze_document(combined_results)
        print(f"   Summary: {analysis.summary}")
        print(f"   Sentiment: {analysis.sentiment}")
        print(f"   Confidence: {analysis.confidence}\n")
        
        return {
            "plan": plan,
            "results": results,
            "analysis": analysis
        }

# Example usage
async def main():
    orchestrator = AgentOrchestrator()
    
    # Example 1: Document Analysis
    print("=" * 60)
    print("Example 1: Document Analysis")
    print("=" * 60 + "\n")
    
    sample_doc = """
    Our company is expanding operations in Casablanca and Rabat.
    We need to hire 50 new employees and establish 2 new offices.
    Budget allocated: 5M MAD. Timeline: 6 months.
    Key challenges: talent acquisition, office space, regulatory compliance.
    """
    
    analysis = await orchestrator.analyze_document(sample_doc)
    print(f"Summary: {analysis.summary}")
    print(f"Key Points: {', '.join(analysis.key_points)}")
    print(f"Sentiment: {analysis.sentiment}\n")
    
    # Example 2: Complex Workflow
    print("=" * 60)
    print("Example 2: Multi-Agent Workflow")
    print("=" * 60 + "\n")
    
    workflow_result = await orchestrator.workflow_example(
        "Launch a new AI-powered customer service system for our Moroccan clients"
    )
    
    print("=" * 60)
    print("✅ Workflow Complete!")
    print("=" * 60)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
