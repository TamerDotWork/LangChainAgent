import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit(1)

# 2. Define Tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together."""
    return a * b

tools = [multiply]

# 3. Initialize Model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-002",
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
response = agent_executor.invoke({"input": "What is 55 multiplied by 10?"})
print(f"Answer: {response['output']}")