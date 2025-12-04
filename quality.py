import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# LOGIC: DATA QUALITY AUDITOR CLASS
# ==========================================
class DataQualityAuditor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.DataFrame()
        
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(file_path)
        except Exception as e:
            print(f"Error reading file: {e}")

    def run_profiling(self):
        if self.df.empty: return None
        df = self.df
        
        # Get numerical column stats, handle non-numeric gracefully
        num_stats = df.describe().to_dict() if not df.empty else {}
        
        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "byte_size": int(df.memory_usage(deep=True).sum()),
            "columns": list(df.columns),
            "preview": df.head(5).to_html(classes='table table-sm table-striped', index=False)
        }

    def check_completeness(self):
        if self.df.empty: return {"score": 0, "null_rate_pct": 0}
        total_cells = self.df.size
        total_missing = self.df.isnull().sum().sum()
        null_rate = (total_missing / total_cells)
        return {"score": (1 - null_rate) * 100, "null_rate_pct": null_rate * 100}

    def check_uniqueness(self, pk_col='id'):
        if self.df.empty: return {"score": 0, "duplicate_rows": 0, "pk_collisions": 0}
        
        total_rows = len(self.df)
        dup_rows = self.df.duplicated().sum()
        row_uniq_rate = 1 - (dup_rows / total_rows)
        
        if pk_col in self.df.columns:
            pk_dups = self.df[pk_col].duplicated().sum()
            pk_uniq_rate = 1 - (pk_dups / total_rows)
            pk_msg = pk_dups
        else:
            pk_uniq_rate = row_uniq_rate 
            pk_msg = "PK Not Found"
            
        avg_score = ((row_uniq_rate + pk_uniq_rate) / 2) * 100
        return {"score": avg_score, "duplicate_rows": int(dup_rows), "pk_collisions": pk_msg}

    def check_validity(self):
        if self.df.empty: return {"score": 0, "negatives": 0}
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        
        total_checks = 0
        passed = 0
        
        for col in numeric_cols:
            n_rows = len(self.df)
            n_neg = (self.df[col] < 0).sum()
            total_checks += n_rows
            passed += (n_rows - n_neg)
            
        score = 100.0 if total_checks == 0 else (passed / total_checks) * 100
        return {"score": score, "negative_value_count": int(total_checks - passed)}

    def check_consistency(self):
        if self.df.empty: return {"score": 0, "issues": 0}
        str_cols = self.df.select_dtypes(include='object').columns
        
        total = 0
        clean = 0
        
        for col in str_cols:
            orig = self.df[col].astype(str)
            trimmed = orig.str.strip()
            total += len(self.df)
            clean += (orig == trimmed).sum()
            
        score = 100.0 if total == 0 else (clean / total) * 100
        return {"score": score, "whitespace_issues": int(total - clean)}

    def get_report(self, pk_col='id'):
        if self.df.empty: return None
        
        weights = {"completeness": 0.25, "uniqueness": 0.25, "validity": 0.25, "consistency": 0.25}
        
        c = self.check_completeness()
        u = self.check_uniqueness(pk_col)
        v = self.check_validity()
        cons = self.check_consistency()
        
        overall = (
            (c['score'] * 0.25) + (u['score'] * 0.25) + 
            (v['score'] * 0.25) + (cons['score'] * 0.25)
        )
        
        return {
            "overall_score": round(overall, 2),
            "profile": self.run_profiling(),
            "details": {
                "completeness": c,
                "uniqueness": u,
                "validity": v,
                "consistency": cons
            }
        }

# ==========================================
# FLASK ROUTES
# ==========================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Route 1: The Upload Page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route 2: The Dashboard Page (Frontend will redirect here)
@app.route('/explore', methods=['GET'])
def explore():
    return render_template('explore.html')

@app.route('/api', methods=['GET', 'POST'])
def api():
    if request.method == 'POST':
        # Check if file is present
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        pk_column = request.form.get('pk_column', 'id') # Get PK name from form
        
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process Data
            auditor = DataQualityAuditor(filepath)
            report = auditor.get_report(pk_col=pk_column)
            
            # Cleanup (Optional: remove file after processing)
            os.remove(filepath)
            
            if report:
                return render_template('explore.html', report=report, filename=filename)
            else:
                return "Error processing file (Empty or Invalid Format)", 400

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=False, use_reloader=False)