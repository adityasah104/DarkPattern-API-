from flask import Flask, request, jsonify, send_from_directory, render_template
import joblib
from bs4 import BeautifulSoup
import requests
import os
from flask_cors import CORS

# --- Initialize Flask app ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # optional: enable if you plan to call API from another domain

# --- Load the ML model ---
MODEL_PATH = 'dark_pattern_model.joblib'
model = joblib.load(MODEL_PATH)


# --- Function: predict dark pattern ---
def predict_dark_pattern(text):
    prediction = model.predict([text])
    return '1' if prediction[0] == 1 else '0'


# --- Function: extract text from webpage ---
def get_text_from_url(url):
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/58.0.3029.110 Safari/537.3'
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text, None
    except requests.Timeout as e:
        return None, f"Request timed out: {e}"
    except requests.RequestException as e:
        return None, f"An error occurred: {e}"


# --- API Route: /predict ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" in request body'}), 400

    url = data['url']
    text, error = get_text_from_url(url)
    if error:
        return jsonify({'error': error}), 500

    prediction = predict_dark_pattern(text)
    return jsonify({
        'prediction': prediction,
        'snippet': text[:1000]  # send part of the text for debugging
    })


# --- Route: Serve Frontend ---
@app.route('/')
def index():
    # If using templates folder (templates/index.html)
    return render_template('index.html')


# --- Optional: Serve static assets (CSS, JS) ---
@app.route('/<path:path>')
def static_proxy(path):
    static_path = os.path.join(app.static_folder, path)
    if os.path.exists(static_path):
        return send_from_directory(app.static_folder, path)
    return "404 Not Found", 404


# --- Main entry point ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
