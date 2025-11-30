from __future__ import annotations
import json
import io
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

dataset = "data.csv"

# Route 1: The Upload Page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route 2: The Dashboard Page (Frontend will redirect here)
@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

# Route 3: The API to Process the file
@app.route('/upload', methods=['POST'])
def upload():
    return jsonify({'status': 'success', 'message': 'File processed successfully'})

# Route 3: The API to Process the file
@app.route('/api', methods=['GET'])
def api():
     
    df = pd.read_csv(dataset)

    dtype_counts = df.dtypes.value_counts()
    most_frequent_type = dtype_counts.idxmax().name if hasattr(dtype_counts.idxmax(), 'name') else str(dtype_counts.idxmax())

    # ---- Invalid fields ----
    invalid_fields = {}
    for col in df.columns:
        # Missing values
        missing_count = df[col].isna().sum()
        
        non_numeric_invalid = 0
        # Try detect non-numeric in numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_invalid = df[col].apply(lambda x: isinstance(x, str)).sum()

        invalid_fields[col] = int(non_numeric_invalid)

    # ---- PII Detection (Basic) ----
    pii_keywords = ["name", "email", "phone", "address", "id", "ssn"]
    pii_fields = [
        col for col in df.columns
        if any(keyword in col.lower() for keyword in pii_keywords)
    ]

     # ---- Duplicate Rows ----
    duplicate_rows = df[df.duplicated()]
    duplicate_count = duplicate_rows.shape[0]

    return jsonify({
        "row_count": df.shape[0],
        "column_count": df.shape[1],
        "most_frequent_dtype": str(most_frequent_type),
        "missing_count": missing_count,

        "pii_fields": pii_fields,
        "invalid_fields": invalid_fields,

    })
    return jsonify({'status': 'success', 'message': dataset.capitalize() + ' dataset loaded successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)