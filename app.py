import os
import json
import uuid
import requests
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# CONFIGURATION
# We will store the results in a folder named 'data_store'
DATA_FOLDER = 'data_store'


# Ensure the storage folder exists when app starts
os.makedirs(DATA_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400

    if file:
        try:
            # Prepare file for API
            file.stream.seek(0)
            files = {'csv_file': (file.filename, file.stream, file.content_type)}

            # 1. Send to External API
            response = requests.post(EXTERNAL_API_URL, files=files)
            response.raise_for_status()
            
            api_data = response.json()

            # 2. Save result to a JSON File (The "No Database" persistence)
            result_id = str(uuid.uuid4())
            filename = f"{result_id}.json"
            filepath = os.path.join(DATA_FOLDER, filename)
            
            with open(filepath, 'w') as f:
                json.dump(api_data, f)

            # 3. Redirect to dashboard with the ID
            return redirect(url_for('dashboard', result_id=result_id))

        except requests.exceptions.RequestException as e:
            return f"API Error: {e}", 500
        except IOError as e:
            return f"File Save Error: {e}", 500

@app.route('/dashboard/<result_id>')
def dashboard(result_id):
    # Construct filepath
    filename = f"{result_id}.json"
    filepath = os.path.join(DATA_FOLDER, filename)
    
    # 4. Check if file exists and read it
    if not os.path.exists(filepath):
        return "Results not found or expired.", 404
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return render_template('dashboard.html', data=data)
    except Exception as e:
        return f"Error reading data: {e}", 500
 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)