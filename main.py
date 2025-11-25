import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
# 1. Import Any for the recursive type hint
from typing import Any, Dict
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit(1)

# --- THE ROOT CAUSE FIX ---
def remove_titles(schema: Dict[str, Any], model: Any) -> None:
    """
    Recursively remove the 'title' key from Pydantic V2 JSON Schemas.
    Google Gemini throws warnings if 'title' is present in tools.
    """
    # Remove title from the current level
    schema.pop("title", None)
    
    # Recursively remove titles from properties (args like 'a' and 'b')
    properties = schema.get("properties", {})
    for prop in properties.values():
        prop.pop("title", None)
        # If you had deeper nesting, you would recurse here, 
        # but for simple tools, this depth is usually sufficient.

class MultiplyArgs(BaseModel):
    a: int = Field(description="The first integer")
    b: int = Field(description="The second integer")

    # Apply the recursive cleaner
    model_config = {
        "json_schema_extra": remove_titles
    }

# Apply the strict schema to the tool
@tool(args_schema=MultiplyArgs)
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together."""
    return a * b

tools = [multiply]

# 3. Initialize Model (Using 1.5-flash-001 for stability)
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0
)

# 4. Create Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the tools provided."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Run
print("--- Starting Agent ---")
response = agent_executor.invoke({"input": "What is 55 multiplied by 10?"})
print(f"Answer: {response['output']}")