import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv

load_dotenv()

# We point to LiteLLM Proxy
# In production, this would be your LiteLLM endpoint
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-1234")

# Initialize the model via LiteLLM
from pydantic_ai.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    base_url=f"{LITELLM_URL}/v1",
    api_key=LITELLM_KEY,
)

model = OpenAIModel(
    'internal-llm',
    provider=provider
)

# Define a simple agent
agent = Agent(
    model,
    system_prompt="You are a professional enterprise assistant for a large Moroccan company. Answer efficiently.",
)

async def main():
    result = await agent.run("What are the advantages of self-hosting LLMs for our organization?")
    print(result.data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
