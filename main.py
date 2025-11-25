import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# ======================================================
# 1. ENV + Setup
# ======================================================
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    print("❌ GOOGLE_API_KEY missing in .env")
    sys.exit(1)

csv_file = "data.csv"

# ======================================================
# 2. Dummy CSV Generator
# ======================================================
if not os.path.exists(csv_file):
    print("⚠️ data.csv not found. Creating dummy dataset...")

    data = {
        "ID": [101, 102, 103, 104, 105, 106, 107, 108],
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hank"],
        "Age": [25, 30, 35, 40, 22, 28, 26, 100],       # Outlier 100
        "Salary": [50000, 60000, 70000, 80000, 45000, 50000, 52000, 500000],  # Skew + outlier
        "Score": [10, 12, 11, 13, 10, 11, 11, 12],
        "Department": ["HR", "IT", "IT", "Sales", "HR", "Sales", "IT", "CEO"]
    }

    pd.DataFrame(data).to_csv(csv_file, index=False)

# Load Data
df = pd.read_csv(csv_file)

# ======================================================
# 3. Automatic Data Health Scan
# ======================================================

def data_health_scan(df):
    print("\n" + "="*60)
    print("📌 DATA HEALTH CHECK")
    print("="*60)

    # --------------------------------------------------
    # 1. Missing Values
    # --------------------------------------------------
    print("\n[1] Missing Values:")
    missing = df.isna().sum()

    if missing.sum() == 0:
        print("   ✅ No missing values found.")
    else:
        for col, val in missing.items():
            if val > 0:
                print(f"   ⚠️ Column '{col}': {val} missing value(s)")

    # --------------------------------------------------
    # 2. Duplicate Rows
    # --------------------------------------------------
    print("\n[2] Duplicate Rows:")
    dup_count = df.duplicated().sum()

    if dup_count == 0:
        print("   ✅ No duplicate rows found.")
    else:
        print(f"   ⚠️ Found {dup_count} duplicate row(s)")

    # --------------------------------------------------
    # 3. Invalid Data
    # --------------------------------------------------
    print("\n[3] Invalid Data:")
    invalid_reported = False

    # Invalid age (expected 0–120)
    if "Age" in df.columns:
        invalid_age = df[(df["Age"] < 0) | (df["Age"] > 120)]
        if len(invalid_age) > 0:
            invalid_reported = True
            print(f"   ⚠️ Invalid Age values (0–120 expected): {len(invalid_age)}")

    # Invalid salary (expected > 0)
    if "Salary" in df.columns:
        invalid_salary = df[(df["Salary"] <= 0)]
        if len(invalid_salary) > 0:
            invalid_reported = True
            print(f"   ⚠️ Invalid Salary values (must be > 0): {len(invalid_salary)}")

    # Text columns with numeric-only content
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        invalid_text = df[df[col].astype(str).str.match(r"^\d+$", na=False)]
        if len(invalid_text) > 0:
            invalid_reported = True
            print(f"   ⚠️ Text column '{col}' contains numeric-only values: {len(invalid_text)}")

    if not invalid_reported:
        print("   ✅ No invalid data detected.")

    print("\n" + "="*60 + "\n")


# ======================================================
# 4. Advanced Scan (Outliers, Skew, Correlation)
# ======================================================

def auto_analyze_data(df):
    print("\n" + "="*60)
    print("📊 AUTOMATIC DATA SCAN")
    print("="*60)

    numeric_cols = df.select_dtypes(include=['number']).columns

    # ------------------ Outliers ------------------
    print("\n[1] Outlier Detection (IQR):")
    outliers = False
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        count = len(df[(df[col] < lower) | (df[col] > upper)])
        if count > 0:
            outliers = True
            print(f"   ⚠️ '{col}': {count} outlier(s) [{lower:.2f} to {upper:.2f}]")

    if not outliers:
        print("   ✅ No outliers found.")

    # ------------------ Skewness ------------------
    print("\n[2] Skewness:")
    skew_vals = df[numeric_cols].skew()
    skewed = False
    for col, v in skew_vals.items():
        if abs(v) > 1:
            skewed = True
            print(f"   ⚠️ '{col}' skewed ({v:.2f})")
    if not skewed:
        print("   ✅ Distributions look normal.")

    # ------------------ Correlation ------------------
    print("\n[3] Strong Correlations (|r| > 0.5):")
    corr = df[numeric_cols].corr()
    pairs = corr.unstack().sort_values(ascending=False)
    pairs = pairs[pairs < 1]
    strong = pairs[abs(pairs) > 0.5].drop_duplicates()

    if strong.empty:
        print("   ℹ️ No strong correlations.")
    else:
        for (c1, c2), v in strong.items():
            print(f"   🔗 {c1} ↔ {c2}: {v:.2f}")

    print("\n" + "="*60 + "\n")


# ======================================================
# 5. Initialize Gemini Flash LLM Agent
# ======================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
    max_output_tokens=2048
)

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    handle_parsing_errors=True
)

# ======================================================
# 6. Main Execution Loop
# ======================================================
if __name__ == "__main__":

    # Run full health checks
    auto_analyze_data(df)
    data_health_scan(df)

    print(f"🤖 Agent Ready! Dataset Loaded with {len(df)} rows.")
    print("Ask me: 'Remove duplicates', 'Show missing values', 'Plot Salary distribution'")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Data Analyst: ")

        if q.lower() in ["exit", "quit", "q"]:
            break

        try:
            agent.invoke({"input": q})
        except Exception as e:
            print(f"❌ Error: {e}")
