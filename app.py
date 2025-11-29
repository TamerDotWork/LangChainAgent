import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent


load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("❌ Error: GOOGLE_API_KEY is missing from .env file.")
    sys.exit(1)

# 2. Configuration & Dummy Data Creation
csv_file = "data.csv"

# We create a file that has EVERY type of error for testing
if not os.path.exists(csv_file):
    print(f"⚠️ {csv_file} not found. Creating dummy data with known issues...")
    data = {
        # High Cardinality (All unique, likely an ID)
        "ID": ["101", "102", "103", "104", "102", "106", "107", "108"], 
        # Whitespace Issue (" Alice") and Missing Value
        "Name": [" Alice", "Bob", "Charlie", "David", "Bob", "Frank ", "Grace", None], 
        # Invalid Data (-5) and Outlier (100)
        "Age": [25, 30, 35, 40, 30, 28, -5, 100],  
        # Skewed Data and Correlation with Age
        "Salary": [50000, 60000, 70000, 80000, 60000, 52000, 50000, 1000000], 
        # Constant Column (Zero Variance)
        "Country": ["USA", "USA", "USA", "USA", "USA", "USA", "USA", "USA"],
        # Mixed types (Numeric but one string makes it Object)
        "Score": [10, 12, 11, 13, 12, 11, "ERROR", 10] 
    }
    # Note: Row 4 (Index 4) is a Duplicate of Row 1 based on Name/Age/Salary/Country logic
    pd.DataFrame(data).to_csv(csv_file, index=False)

# 3. Load Data
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    print(f"❌ Could not read file: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# COMPREHENSIVE DATA HEALTH SCAN
# ---------------------------------------------------------
def auto_analyze_data(dataframe):
    print("\n" + "="*60)
    print(f"📊 FULL DATA HEALTH REPORT: {csv_file}")
    print("="*60)
    
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    object_cols = dataframe.select_dtypes(include=['object']).columns

    # --- BASIC CHECKS ---
    
    # [1] Missing Values
    print("\n[1] Missing Values:")
    missing = dataframe.isnull().sum()
    if missing.sum() == 0:
        print("    ✅ No missing values.")
    else:
        print(missing[missing > 0].to_string())

    # [2] Duplicate Rows
    print("\n[2] Duplicate Rows:")
    dupes = dataframe.duplicated().sum()
    if dupes == 0:
        print("    ✅ No duplicate rows.")
    else:
        print(f"    ⚠️ Found {dupes} duplicate row(s).")

    # [3] Invalid Data (Negative Numbers)
    print("\n[3] Logic Check (Negative Numbers):")
    neg_found = False
    for col in numeric_cols:
        neg_count = (dataframe[col] < 0).sum()
        if neg_count > 0:
            neg_found = True
            print(f"    ⚠️ Column '{col}' has {neg_count} negative value(s).")
    if not neg_found:
        print("    ✅ No negative numbers found.")

    # --- ADVANCED CHECKS ---

    # [4] Constant Columns (Zero Variance)
    print("\n[4] Zero Variance (Useless Columns):")
    constant_cols = [col for col in dataframe.columns if dataframe[col].nunique() <= 1]
    if constant_cols:
        for col in constant_cols:
            val = dataframe[col].iloc[0] if len(dataframe) > 0 else "Empty"
            print(f"    ⚠️ Column '{col}' has only 1 unique value: '{val}'.")
    else:
        print("    ✅ No constant columns.")

    # [5] String Hygiene (Whitespace)
    print("\n[5] Whitespace Check (Dirty Strings):")
    whitespace_found = False
    for col in object_cols:
        # Check if column contains strings and if they have extra spaces
        if dataframe[col].dtype == object:
            # Safely check strings, skipping NaNs
            dirty_count = dataframe[col].apply(lambda x: isinstance(x, str) and len(x) != len(x.strip())).sum()
            if dirty_count > 0:
                whitespace_found = True
                print(f"    ⚠️ Column '{col}' has {dirty_count} values with leading/trailing spaces.")
    
    if not whitespace_found:
        print("    ✅ Text looks clean.")

    # [6] Outliers (IQR Method)
    print("\n[6] Outlier Detection (IQR):")
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
            print(f"    ⚠️ Column '{col}': {count} outlier(s).")
    
    if not outliers_found:
        print("    ✅ No significant outliers.")

    # [7] Skewness & Correlation
    print("\n[7] Statistical Insights:")
    # Skew
    skew_found = False
    skew_vals = dataframe[numeric_cols].skew()
    for col, val in skew_vals.items():
        if abs(val) > 1.0:
            skew_found = True
            print(f"    ⚠️ Column '{col}' is highly skewed ({val:.2f}).")
    if not skew_found:
        print("    ✅ Distributions look normal.")
    
    # Correlation
    corr_matrix = dataframe[numeric_cols].corr()
    pairs = corr_matrix.unstack().sort_values(ascending=False)
    pairs = pairs[pairs < 1.0] # Remove self
    strong_pairs = pairs[abs(pairs) > 0.5].drop_duplicates()
    if len(strong_pairs) > 0:
        for (col1, col2), val in strong_pairs.items():
            print(f"    🔗 Correlation: {col1} vs {col2} ({val:.2f})")

    print("\n" + "="*60 + "\n")
# ---------------------------------------------------------

# 4. Initialize Model
# Using the model ID for the latest Flash version
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

    print(f"🤖 Data Agent Ready! ({len(df)} rows loaded)")
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