"""
Flask Web App for Customer Support Ticket Classifier
"""
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from classifier import classify
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE = Path(__file__).parent

# Load the dashboard HTML
with open(BASE.parent / 'SupportAI_Dashboard.html', 'r', encoding='utf-8') as f:
    DASHBOARD_HTML = f.read()

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/classify', methods=['POST'])
def api_classify():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        result = classify(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics')
def api_metrics():
    try:
        with open(BASE / 'model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)