# api.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from main import DataQualityEngine
import pandas as pd

app = FastAPI(title="Data Quality Engine API")

# Demo dataset for GET requests
demo_data = {
    "ID": [1, 2, 3, 4, 4],
    "Name": ["Alice", "Bob", "12345", "David", "David"],
    "Age": [25, 30, -5, 200, 200],
    "Salary": [50000, 0, 70000, 80000, 80000],
    "Score": [10, 12, 11, None, None],
    "Joined": ["2021-01-01", "2020-06-05", "not_a_date", "2022-03-03", None]
}
df_demo = pd.DataFrame(demo_data)

# Optional: custom rule example
def score_invalid_mask(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.isna() | (s < 0) | (s > 100)

@app.get("/api/")
def get_data_quality():
    engine = DataQualityEngine(
        df_demo,
        name="demo_dataset",
        validation_rules={"Score": score_invalid_mask},
        sample_size=3
    )
    report = engine.run_all()
    return JSONResponse(content=report)
