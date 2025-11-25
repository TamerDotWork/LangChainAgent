"""
main.py

Generic Data Quality Engine for pandas DataFrames + FastAPI API.

Requirements:
    pandas, numpy, scipy (optional for zscore), fastapi, uvicorn, python-multipart,
    langchain-google-genai, langchain-experimental
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
import pandas as pd
import numpy as np

# Optional import; zscore fallback if scipy not available
try:
    from scipy.stats import zscore
except Exception:
    zscore = None

# LangChain imports (optional)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io

# ------------------------
# Data Quality Engine
# ------------------------
@dataclass
class IssueSample:
    rows: List[Dict[str, Any]] = field(default_factory=list)


class DataQualityEngine:
    """Generic Data Quality Engine (full code as provided previously)."""

    def __init__(
        self,
        df: pd.DataFrame,
        name: str = "dataset",
        validation_rules: Optional[Dict[str, Callable[[pd.Series], pd.Series]]] = None,
        sample_size: int = 5,
    ):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        self.df = df.copy()
        self.name = name
        self.sample_size = sample_size
        self.validation_rules = validation_rules or {}
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.text_cols = self.df.select_dtypes(include=["object", "string"]).columns.tolist()
        self.datetime_cols = self._detect_datetime_columns()

    def _detect_datetime_columns(self) -> List[str]:
        dt_cols = []
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                dt_cols.append(col)
                continue
            try:
                sample = self.df[col].dropna().astype(str).head(10)
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= max(1, len(sample) // 2):
                    dt_cols.append(col)
            except Exception:
                pass
        return dt_cols

    def _sample_rows(self, mask: pd.Series) -> List[Dict[str, Any]]:
        try:
            rows = self.df.loc[mask].head(self.sample_size).to_dict(orient="records")
        except Exception:
            rows = []
        return rows

    # ------------------------
    # Scans
    # ------------------------
    def scan_missing(self) -> Dict[str, Any]:
        missing_counts = self.df.isna().sum().to_dict()
        total_missing = int(self.df.isna().sum().sum())
        percent_missing_by_col = (self.df.isna().mean() * 100).round(3).to_dict()

        details = {}
        for col, cnt in missing_counts.items():
            if cnt > 0:
                mask = self.df[col].isna()
                details[col] = {
                    "missing_count": int(cnt),
                    "missing_pct": float(percent_missing_by_col[col]),
                    "sample_rows": self._sample_rows(mask)
                }

        return {
            "metric": "completeness",
            "total_missing": total_missing,
            "missing_by_column": missing_counts,
            "missing_by_column_pct": percent_missing_by_col,
            "details": details
        }

    def scan_duplicates(self) -> Dict[str, Any]:
        dup_mask = self.df.duplicated(keep="first")
        dup_count = int(dup_mask.sum())
        dup_samples = self._sample_rows(dup_mask)
        dup_summary = {}
        if dup_count > 0:
            grouped = (
                self.df[dup_mask]
                .astype(str)
                .apply(lambda r: "|".join(r.values.tolist()), axis=1)
                .value_counts()
                .head(10)
                .to_dict()
            )
            dup_summary = grouped
        return {
            "metric": "uniqueness",
            "duplicate_count": dup_count,
            "duplicate_samples": dup_samples,
            "duplicate_summary_top": dup_summary
        }

    def scan_schema_types(self) -> Dict[str, Any]:
        dtype_map = {col: str(self.df[col].dtype) for col in self.df.columns}
        parsed_dates = self.datetime_cols
        return {
            "metric": "schema",
            "dtypes": dtype_map,
            "datetime_columns_detected": parsed_dates
        }

    def scan_invalid(self) -> Dict[str, Any]:
        invalid_report = {}
        total_invalid = 0

        if "Age" in self.df.columns:
            mask = (pd.to_numeric(self.df["Age"], errors="coerce").isna()) | (self.df["Age"] < 0) | (self.df["Age"] > 120)
            count = int(mask.sum())
            if count > 0:
                invalid_report["Age"] = {
                    "count": count,
                    "reason": "Age should be numeric and between 0 and 120",
                    "sample_rows": self._sample_rows(mask)
                }
                total_invalid += count

        if "Salary" in self.df.columns:
            mask = pd.to_numeric(self.df["Salary"], errors="coerce").isna() | (self.df["Salary"] <= 0)
            count = int(mask.sum())
            if count > 0:
                invalid_report["Salary"] = {
                    "count": count,
                    "reason": "Salary should be numeric and > 0",
                    "sample_rows": self._sample_rows(mask)
                }
                total_invalid += count

        for col in self.text_cols:
            mask = self.df[col].astype(str).str.match(r"^\d+$", na=False)
            count = int(mask.sum())
            if count > 0:
                invalid_report[col] = {
                    "count": count,
                    "reason": "Text column contains numeric-only values",
                    "sample_rows": self._sample_rows(mask)
                }
                total_invalid += count

        for col, rule_fn in self.validation_rules.items():
            if col not in self.df.columns:
                continue
            try:
                mask = rule_fn(self.df[col])
                if not isinstance(mask, pd.Series):
                    continue
                count = int(mask.sum())
                if count > 0:
                    invalid_report[f"custom:{col}"] = {
                        "count": count,
                        "reason": "Custom validation rule triggered",
                        "sample_rows": self._sample_rows(mask)
                    }
                    total_invalid += count
            except Exception as e:
                invalid_report[f"custom_error:{col}"] = {
                    "count": 0,
                    "reason": f"custom rule error: {e}"
                }

        return {
            "metric": "validity",
            "total_invalid": total_invalid,
            "invalid_by_field": invalid_report
        }

    def scan_outliers(self, method: str = "iqr", iqr_multiplier: float = 1.5, zscore_threshold: float = 3.0) -> Dict[str, Any]:
        result = {"metric": "outliers", "method": method, "columns": {}}
        for col in self.numeric_cols:
            series = self.df[col].dropna().astype(float)
            if series.empty:
                continue
            masks = []
            if method in ("iqr", "both"):
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr
                mask_iqr = (self.df[col] < lower) | (self.df[col] > upper)
                masks.append(("iqr", mask_iqr, {"lower": float(lower), "upper": float(upper)}))
            if method in ("zscore", "both"):
                if zscore is not None:
                    scores = zscore(series)
                    z_mask = pd.Series(False, index=self.df.index)
                    z_mask.loc[series.index] = np.abs(scores) > zscore_threshold
                    masks.append(("zscore", z_mask, {"threshold": zscore_threshold}))
            combined_mask = pd.Series(False, index=self.df.index)
            details = {}
            for name, mask, meta in masks:
                combined_mask = combined_mask | mask
                details[name] = {
                    "count": int(mask.sum()),
                    "meta": meta,
                    "sample_rows": self._sample_rows(mask)
                }
            total = int(combined_mask.sum())
            if total > 0:
                result["columns"][col] = {
                    "outlier_count": total,
                    "details": details,
                    "sample_rows": self._sample_rows(combined_mask)
                }
        return result

    def scan_skewness(self) -> Dict[str, Any]:
        skew_vals = {}
        for col in self.numeric_cols:
            try:
                val = float(self.df[col].skew(skipna=True))
                skew_vals[col] = round(val, 6)
            except Exception:
                skew_vals[col] = None
        heavy = {c: v for c, v in skew_vals.items() if v is not None and abs(v) > 1.0}
        return {
            "metric": "skewness",
            "skew_values": skew_vals,
            "heavy_skew_columns": heavy
        }

    def scan_correlation(self, threshold: float = 0.5, method: str = "pearson") -> Dict[str, Any]:
        if not self.numeric_cols or len(self.numeric_cols) < 2:
            return {"metric": "correlation", "pairs": {}}
        corr = self.df[self.numeric_cols].corr(method=method)
        pairs = {}
        cols = corr.columns.tolist()
        for i, c1 in enumerate(cols):
            for j in range(i+1, len(cols)):
                c2 = cols[j]
                val = corr.at[c1, c2]
                if pd.isna(val):
                    continue
                if abs(val) >= threshold:
                    pairs[f"{c1}__{c2}"] = round(float(val), 6)
        return {"metric": "correlation", "method": method, "threshold": threshold, "pairs": pairs}

    def scan_text_quality(self) -> Dict[str, Any]:
        report = {}
        for col in self.text_cols:
            ser = self.df[col].astype(str).replace({"nan": None})
            non_null = ser.dropna()
            if non_null.empty:
                continue
            lengths = non_null.map(len)
            numeric_only = non_null.str.match(r"^\d+$", na=False).sum()
            distinct = int(non_null.nunique(dropna=True))
            avg_len = float(lengths.mean())
            report[col] = {
                "avg_length": round(avg_len, 3),
                "numeric_only_count": int(numeric_only),
                "distinct_count": distinct,
                "missing_pct": float(1 - len(non_null) / len(self.df)) * 100
            }
        return {"metric": "text_quality", "columns": report}

    def data_quality_score(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        w = weights or {
            "completeness": 0.35,
            "validity": 0.25,
            "uniqueness": 0.15,
            "schema": 0.15,
            "text_quality": 0.10
        }
        total_cells = self.df.size
        total_missing = int(self.df.isna().sum().sum())
        completeness_pct = (1 - total_missing / total_cells) * 100 if total_cells > 0 else 100.0
        invalid = self.scan_invalid().get("total_invalid", 0)
        validity_pct = max(0.0, (1 - invalid / max(1, len(self.df))) * 100)
        dup_count = int(self.df.duplicated().sum())
        uniqueness_pct = (1 - dup_count / max(1, len(self.df))) * 100
        dtype_score = 100.0
        num_obj_numeric = sum(1 for c in self.numeric_cols if self.df[c].dtype == object)
        if num_obj_numeric:
            dtype_score -= min(20, 5 * num_obj_numeric)
        text_q = self.scan_text_quality()
        numeric_only_total = sum(v.get("numeric_only_count", 0) for v in text_q.get("columns", {}).values())
        text_quality_pct = max(0.0, (1 - numeric_only_total / max(1, len(self.df))) * 100)
        score = (
            completeness_pct * w.get("completeness", 0) +
            validity_pct * w.get("validity", 0) +
            uniqueness_pct * w.get("uniqueness", 0) +
            dtype_score * w.get("schema", 0) +
            text_quality_pct * w.get("text_quality", 0)
        )
        return {
            "metric": "data_quality_score",
            "score": round(float(score), 3),
            "components": {
                "completeness_pct": round(float(completeness_pct), 3),
                "validity_pct": round(float(validity_pct), 3),
                "uniqueness_pct": round(float(uniqueness_pct), 3),
                "schema_pct": round(float(dtype_score), 3),
                "text_quality_pct": round(float(text_quality_pct), 3)
            },
            "weights": w
        }

    def run_all(self, outlier_method: str = "iqr", correlation_threshold: float = 0.5) -> Dict[str, Any]:
        result = {
            "dataset_name": self.name,
            "row_count": int(len(self.df)),
            "column_count": int(len(self.df.columns)),
            "scans": {}
        }
        result["scans"]["schema"] = self.scan_schema_types()
        result["scans"]["missing"] = self.scan_missing()
        result["scans"]["duplicates"] = self.scan_duplicates()
        result["scans"]["invalid"] = self.scan_invalid()
        result["scans"]["outliers"] = self.scan_outliers(method=outlier_method)
        result["scans"]["skewness"] = self.scan_skewness()
        result["scans"]["correlation"] = self.scan_correlation(threshold=correlation_threshold)
        result["scans"]["text_quality"] = self.scan_text_quality()
        result["scans"]["quality_score"] = self.data_quality_score()
        return result

    def to_json(self, result: Dict[str, Any], path: Optional[str] = None) -> str:
        s = json.dumps(result, default=self._json_fallback, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    @staticmethod
    def _json_fallback(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if pd.isna(o):
            return None
        try:
            return str(o)
        except Exception:
            return None

    def create_langchain_agent(self, llm_model: str = "gemini-flash-latest", temperature: float = 0.0, max_output_tokens: int = 1024, verbose: bool = True):
        llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature, max_output_tokens=max_output_tokens)
        agent = create_pandas_dataframe_agent(
            llm,
            self.df,
            verbose=verbose,
            allow_dangerous_code=False,
            handle_parsing_errors=True
        )
        return agent


# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Data Quality API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo dataset and engine
demo_data = {
    "ID": [1, 2, 3, 4, 4],
    "Name": ["Alice", "Bob", "12345", "David", "David"],
    "Age": [25, 30, -5, 200, 200],
    "Salary": [50000, 0, 70000, 80000, 80000],
    "Score": [10, 12, 11, None, None],
    "Joined": ["2021-01-01", "2020-06-05", "not_a_date", "2022-03-03", None]
}
df_demo = pd.DataFrame(demo_data)
def score_invalid_mask(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.isna() | (s < 0) | (s > 100)
engine_demo = DataQualityEngine(df_demo, name="demo_dataset", validation_rules={"Score": score_invalid_mask}, sample_size=3)

@app.get("/")
def root():
    return {"message": "Data Quality API running. Use GET /api or POST /api with CSV/JSON upload."}

@app.get("/api")
def get_demo_report():
    return engine_demo.run_all()

@app.post("/api")
async def post_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif file.filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or JSON.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    
    # Run engine dynamically
    engine = DataQualityEngine(df, name=file.filename, sample_size=3)
    return engine.run_all()

# ------------------------
# Run uvicorn if main
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5006, reload=True)
