from flask import Flask, request, jsonify
import joblib
from bs4 import BeautifulSoup
import requests

# Define the Flask application
app = Flask(__name__)

# Load the model (Assuming the model is in the same directory as this script)
model = joblib.load('dark_pattern_model.joblib')









    # Prediction function
def predict_dark_pattern(text):
    prediction = model.predict([text])
    return "Dark Pattern" if prediction[0] == 1 else "Not Dark Pattern"




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
    # else:
    #     return jsonify({'error': 'Unable to extract text from the provided URL.'}), 500
    

# Function to extract text from URL
def get_text_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        # if response.status_code != 200:
        #     return None, f"Error fetching the webpage: Status code {response.status_code}"
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ')
        return text, None
    except requests.Timeout as e:
        return None, f"The request timed out: {e}"
    except requests.RequestException as e:
        return None, f"An error occurred: {e}"




# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)