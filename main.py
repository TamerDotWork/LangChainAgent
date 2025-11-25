import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
# 1. NEW: Import Pydantic for the fix
from pydantic import BaseModel, Field

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit(1)

# 2. NEW: Define the explicit Schema to fix the "title" warning
class MultiplyArgs(BaseModel):
    a: int = Field(description="The first integer to multiply")
    b: int = Field(description="The second integer to multiply")

# 3. Apply the schema to the tool
@tool(args_schema=MultiplyArgs)
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together."""
    return a * b

tools = [multiply]

# 4. Initialize Model
# Note: 'gemini-1.5-flash-001' is often more stable for tools than 'latest'
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0
)

# 5. Create Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the tools provided."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Run
response = agent_executor.invoke({"input": "What is 55 multiplied by 10?"})
print(f"Answer: {response['output']}")