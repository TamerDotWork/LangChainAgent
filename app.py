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


 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)