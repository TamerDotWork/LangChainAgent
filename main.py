import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. Setup
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY missing.")
    exit(1)

# 2. Create Dummy Data (So you have something to analyze)
csv_file = "data.csv"
 

# 3. Load Data
df = pd.read_csv(csv_file)

# 4. Initialize Model
# "gemini-flash-latest" points to the latest stable flash version
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0
)

# 5. Create Agent
# allow_dangerous_code=True allows the AI to write Python to solve your math
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True
)

# 6. Run Loop
if __name__ == "__main__":
    print(f"📊 Agent ready! Loaded {len(df)} rows from {csv_file}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            q = input("Data Analyst: ")
            if q.lower() in ["exit", "quit"]:
                break
            
            # Run the agent
            agent.invoke({"input": q})
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")