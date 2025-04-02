from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gdown

app = Flask(__name__)
CORS(app)

# Google Drive File ID
file_id = "1qz2FthNmqPBf2utHCxfPB9ZfSNrhxavK"
url = f"https://drive.google.com/uc?id=1qz2FthNmqPBf2utHCxfPB9ZfSNrhxavK"

# Download model if not already downloaded
output = "crop_yield_model.pkl"
gdown.download(url, output, quiet=False)

# Load the model
model = joblib.load(output)

# Assuming the 'Item' column values in your dataset
item_labels = ['Potatoes', 'Maize', 'Wheat', 'Rice']  # Add all possible items here

# Initialize the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(item_labels)  # Fit the encoder with the items

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML file when accessing the root URL

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Encode 'Item' as a numeric value
    item_encoded = label_encoder.transform([data['Item']])[0]
    
    # Prepare features
    features = np.array([
        data['Area'],
        item_encoded,  # Using the encoded value
        data['Year'],
        data['average_rain_fall_mm_per_year'],
        data['pesticides_tonnes'],
        data['avg_temp']
    ]).reshape(1, -1)
    
    # Predict yield
    prediction = model.predict(features)[0]
    
    return jsonify({'predicted_yield': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
