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
    return jsonify({
        "row_count": df.shape[0],
        "column_count": df.shape[1],
        "most_frequent_type": most_frequent_type,
    })
    return jsonify({'status': 'success', 'message': dataset.capitalize() + ' dataset loaded successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)