from __future__ import annotations
import json
import io
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
# Optional: Scipy for Z-Score outliers
try:
    from scipy.stats import zscore
except ImportError:
    zscore = None

from flask import Flask, render_template, request, jsonify,url_for
# ------------------------
# Flask Application
# ------------------------
app = Flask(__name__)
# Route 2: The Dashboard Page (Frontend will redirect here)
@app.route('/config', methods=['GET'])
def config():
    return render_template('config.html')

# Route 1: The Upload Page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route 2: The Dashboard Page (Frontend will redirect here)
@app.route('/explore', methods=['GET'])
def explore():
    return render_template('explore.html')

# Route 3: The API to Process the file
@app.route('/api', methods=['POST'])
def api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read file into Pandas
        if file.filename.endswith('.csv'):
            # Convert to string buffer for robust CSV reading
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            df = pd.read_csv(stream)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({"error": "Unsupported file type. Please upload .csv or .json"}), 400
        

        #df = pd.read_csv(dataset)

        # ---- Most frequent dtype ----
        dtype_counts = df.dtypes.value_counts()
        most_frequent_dtype = str(dtype_counts.index[0]) if len(dtype_counts) > 0 else None

        # ---- Invalid fields per column (only include if count > 0) ----
        invalid_fields = {}
        for col in df.columns:
            # Missing values
            missing_count = int(df[col].isna().sum())
            
            # Non-numeric in numeric columns
            non_numeric_invalid = 0
            if pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_invalid = int(df[col].apply(lambda x: isinstance(x, str)).sum())
            
            total_invalid = missing_count + non_numeric_invalid
            if total_invalid > 0:
                invalid_fields[col] = total_invalid

        # ---- PII Detection ----
        pii_keywords = ["name", "email", "phone", "address", "id", "ssn"]
        pii_fields = [
            col for col in df.columns
            if any(keyword in col.lower() for keyword in pii_keywords)
        ]

        # ---- Duplicate Rows ----
        duplicate_count = int(df.duplicated().sum())


        # ---- Distribution & Outliers ----
        distribution = {}
        outliers = {}

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            col_data = df[col].dropna()  # ignore NaNs for stats
            # Distribution stats
            distribution[col] = {
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "std": float(col_data.std()),
                "25%": float(col_data.quantile(0.25)),
                "50%": float(col_data.quantile(0.5)),
                "75%": float(col_data.quantile(0.75))
            }
            
            # Outliers (using Z-score threshold > 3)
            if len(col_data) > 1:
                z_scores = np.abs(zscore(col_data))
                outlier_count = int((z_scores > 3).sum())
                if outlier_count > 0:  # include only if there are outliers
                    outliers[col] = outlier_count

            # ---- Correlation ----
            correlation_threshold = 0.5  # only return correlations above this
            correlation = {}

            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr(method='pearson')
                # Iterate upper triangle to avoid duplicate pairs
                for i, col1 in enumerate(corr_matrix.columns):
                    for j, col2 in enumerate(corr_matrix.columns):
                        if j > i:  # upper triangle
                            corr_value = float(corr_matrix.iloc[i, j])
                            if abs(corr_value) >= correlation_threshold:
                                pair_name = f"{col1}__{col2}"
                                correlation[pair_name] = corr_value

        total_cells = df.shape[0] * df.shape[1]

        # ---- Missing / Invalid Fields Penalty ----
        total_invalid = sum(invalid_fields.values())
        invalid_penalty = total_invalid / total_cells  # fraction of invalid cells

        # ---- Duplicate Rows Penalty ----
        duplicate_penalty = duplicate_count / df.shape[0]  # fraction of duplicate rows

        # ---- Outliers Penalty ----
        total_outliers = sum(outliers.values())
        outlier_penalty = total_outliers / total_cells  # fraction of outlier cells

        # ---- Combine penalties to compute quality score ----
        # Higher penalties → lower score, scale 0-100
        score = 100 * (1 - (invalid_penalty + duplicate_penalty + outlier_penalty))
        score = max(0, min(100, score))  # ensure between 0-100
 
    
        return jsonify({
            "quality_score":int(round(score, 0)),
            "dataset_name": file.filename,

            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "most_frequent_dtype": most_frequent_dtype,
            "preview": df.head(5).to_dict(orient='records'),

            "missing_count": int(missing_count),
            "duplicate_count": int(duplicate_count),
            "invalid_fields": {str(k): int(v) for k, v in invalid_fields.items()},
            "pii_fields": pii_fields,
            "distribution": distribution,
            "outliers": outliers,
            "correlation": correlation
        })
        

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)