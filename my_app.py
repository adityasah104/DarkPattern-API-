from flask import Flask, request, jsonify
import joblib
from bs4 import BeautifulSoup
import requests

# Define the Flask application
app = Flask(__name__)

# Load the model
model = joblib.load('dark_pattern_model.joblib')





    # Prediction function
def predict_dark_pattern(text):
    prediction = model.predict([text])
    return '1' if prediction[0] == 1 else '0'




# Define the predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    json_input = request.json
    url = json_input['url']
    text, error = get_text_from_url(url)
    if error is not None:
        return jsonify({'error': error}), 500
    if text:
        prediction = predict_dark_pattern(text)
        return jsonify({'prediction': prediction})

    

# Function to extract text from URL
def get_text_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ')
        return text, None
    except requests.Timeout as e:
        return None, f"The request timed out: {e}"
    except requests.RequestException as e:
        return None, f"An error occurred: {e}"


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
    
