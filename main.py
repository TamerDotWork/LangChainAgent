import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. Setup
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("❌ Error: GOOGLE_API_KEY is missing from .env file.")
    sys.exit(1)

# 2. Configuration & Dummy Data Generation
csv_file = "data.csv"

# Create specific data to trigger Outliers, Skew, and Correlation checks
if not os.path.exists(csv_file):
    print(f"⚠️ {csv_file} not found. Creating dummy data...")
    data = {
        "ID": [101, 102, 103, 104, 105, 106, 107, 108],
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hank"],
        # Age: 100 is an Outlier
        "Age": [25, 30, 35, 40, 22, 28, 26, 100],  
        # Salary: Highly Skewed (Hank makes way more) and Correlated with Age
        "Salary": [50000, 60000, 70000, 80000, 45000, 50000, 52000, 500000], 
        "Score": [10, 12, 11, 13, 10, 11, 11, 12],
        "Department": ["HR", "IT", "IT", "Sales", "HR", "Sales", "IT", "CEO"]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

# 3. Load Data
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    print(f"❌ Could not read file: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# AUTOMATIC DATA HEALTH SCAN (Outliers, Skew, Correlation)
# ---------------------------------------------------------
def auto_analyze_data(dataframe):
    print("\n" + "="*60)
    print(f"📊 AUTOMATIC DATA SCAN: {csv_file}")
    print("="*60)
    
    numeric_cols = dataframe.select_dtypes(include=['number']).columns

    # [1] Outlier Detection (IQR Method)
    print("\n[1] Outlier Detection (IQR):")
    outliers_found = False
    for col in numeric_cols:
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        count = len(dataframe[(dataframe[col] < lower) | (dataframe[col] > upper)])
        if count > 0:
            outliers_found = True
            print(f"    ⚠️ Column '{col}': {count} outlier(s) found (outside {lower:.1f} - {upper:.1f}).")
    
    if not outliers_found:
        print("    ✅ No significant outliers found.")

    # [2] Skewness Check
    print("\n[2] Skewness Check:")
    skew_found = False
    skew_vals = dataframe[numeric_cols].skew()
    for col, val in skew_vals.items():
        if abs(val) > 1.0:
            skew_found = True
            direction = "Positive (Right)" if val > 0 else "Negative (Left)"
            print(f"    ⚠️ Column '{col}' is highly skewed ({val:.2f}) -> {direction} Tail")
    
    if not skew_found:
        print("    ✅ Distributions look normal.")

    # [3] Pair Correlation (Pearson > 0.5)
    print("\n[3] Strong Correlations:")
    corr_matrix = dataframe[numeric_cols].corr()
    pairs = corr_matrix.unstack().sort_values(ascending=False)
    pairs = pairs[pairs < 1.0] # Remove self-correlation
    strong_pairs = pairs[abs(pairs) > 0.5].drop_duplicates()
    
    if len(strong_pairs) > 0:
        for (col1, col2), val in strong_pairs.items():
            print(f"    🔗 {col1} vs {col2}: {val:.2f}")
    else:
        print("    ℹ️ No strong correlations found.")

    print("\n" + "="*60 + "\n")

# ---------------------------------------------------------

# 4. Initialize Model
# STRICTLY using "gemini-flash-latest" as requested
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
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
    # RUN SCAN
    auto_analyze_data(df)

    print(f"🤖 Agent Ready! ({len(df)} rows)")
    print("Ask: 'Remove outliers in Age' or 'Plot Salary distribution'.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            q = input("Data Analyst: ")
            if q.lower() in ["exit", "quit", "q"]:
                break
            
            agent.invoke({"input": q})
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")