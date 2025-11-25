import os
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. Setup
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("❌ Error: GOOGLE_API_KEY is missing from .env file.")
    sys.exit(1)

# 2. Configuration
csv_file = "data.csv"

# --- HELPER: Create dummy data if file doesn't exist (For testing) ---
if not os.path.exists(csv_file):
    print(f"⚠️ {csv_file} not found. Creating dummy data for testing...")
    data = {
        "ID": [101, 102, 103, 102, 105],           # 102 is Duplicate
        "Name": ["Alice", "Bob", "Charlie", "Bob", "Eve"], 
        "Age": [25, 30, None, 30, -5],             # Missing AND Invalid (-5)
        "Salary": [50000, 80000, 75000, 80000, 90000],
        "Department": ["HR", "IT", "IT", "IT", "HR"]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
# --------------------------------------------------------------------

# 3. Load Data
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    print(f"❌ Could not read file: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# FEATURE: Automatic Data Analysis (Runs on startup)
# ---------------------------------------------------------
def auto_analyze_data(dataframe):
    """Analyzes the dataframe for common data quality issues."""
    print("\n" + "="*50)
    print(f"📊 AUTOMATIC DATA SCAN: {csv_file}")
    print("="*50)
    
    # 1. Count Missing Values
    print("\n[1] Missing Values Check:")
    missing = dataframe.isnull().sum()
    if missing.sum() == 0:
        print("    ✅ No missing values found.")
    else:
        # Filter to show only columns with missing data
        print(missing[missing > 0].to_string())

    # 2. Count Duplicate Rows
    print("\n[2] Duplicate Rows Check:")
    dupes = dataframe.duplicated().sum()
    if dupes == 0:
        print("    ✅ No duplicate rows found.")
    else:
        print(f"    ⚠️ Found {dupes} duplicate row(s).")

    # 3. Count Invalid Data (Negative numbers in numeric columns)
    print("\n[3] Invalid Data Check (Negative Numbers):")
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    issues_found = False
    
    for col in numeric_cols:
        # Check for negative values
        neg_count = (dataframe[col] < 0).sum()
        if neg_count > 0:
            print(f"    ⚠️ Column '{col}' has {neg_count} negative value(s).")
            issues_found = True
            
    if not issues_found:
        print("    ✅ No negative values found in numeric columns.")

    print("\n" + "="*50 + "\n")
# ---------------------------------------------------------

# 4. Initialize Model
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", # Specific stable version
    temperature=0
)

# 5. Create Agent
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    handle_parsing_errors=True
)

# 6. Run Loop
if __name__ == "__main__":
    # RUN THE ANALYSIS AUTOMATICALLY
    auto_analyze_data(df)

    print(f"🤖 Data Agent Ready! ({len(df)} rows loaded)")
    print("You can ask: 'Drop duplicates', 'Fill missing Age', or 'Plot Salary'.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            q = input("Data Analyst: ")
            if q.lower() in ["exit", "quit", "q"]:
                print("Bye!")
                break
            
            # Run the agent
            agent.invoke({"input": q})
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")