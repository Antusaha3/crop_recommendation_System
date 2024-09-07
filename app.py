from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import logging

app = Flask(__name__)

# Load the model and scaler
with open('crop_model.pkl', 'rb') as model_file:
    crop_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Map definitions for converting form inputs
soil_color_map = {
    "Black": 1, "Red": 2, "Dark Brown": 3, "Reddish Brown": 4,
    "Light Brown": 5, "Medium Brown": 6
}

fertilizer_map = {
    "Urea": 1, "DAP": 2, "MOP": 3, "19:19:19 NPK": 4, "SSP": 5,
    "Magnesium Sulphate": 6, "10:26:26 NPK": 7, "50:26:26 NPK": 8,
    "Chelated Micronutrient": 9, "12:32:16 NPK": 10, "Ferrous Sulphate": 11,
    "13:32:26 NPK": 12, "Ammonium Sulphate": 13, "10:10:10 NPK": 14,
    "Hydrated Lime": 15, "White Potash": 16, "20:20:20 NPK": 17,
    "18:46:00 NPK": 18, "Sulphur": 19
}

crop_map = {
    1: "Sugarcane", 2: "Wheat", 3: "Cotton", 4: "Jowar", 5: "Maize",
    6: "Rice", 7: "Groundnut", 8: "Tur", 9: "Grapes", 10: "Ginger",
    11: "Urad", 12: "Moong", 13: "Gram", 14: "Turmeric", 15: "Soybean",
    16: "Masoor", 17: "Banana", 18: "Sunflower", 19: "Pigeon Pea", 20: "Cabbage"
}

fertilizer_names = list(fertilizer_map.keys())
soil_color_names = list(soil_color_map.keys())

@app.route("/")
def home():
    return render_template("index.html", fertilizer_names=fertilizer_names, soil_color_names=soil_color_names)

@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        logging.debug("Predict Crop Request: %s", request.json)

        # Extract data from the form
        soil_color = request.json['soilColor']
        nitrogen = float(request.json['nitrogen'])
        phosphorus = float(request.json['phosphorus'])
        potassium = float(request.json['potassium'])
        ph = float(request.json['ph'])
        rainfall = float(request.json['rainfall'])
        temperature = float(request.json['temperature'])
        fertilizer = request.json['fertilizer']

        # Convert inputs to appropriate numerical values
        soil_color_val = soil_color_map.get(soil_color, 0)
        fertilizer_val = fertilizer_map.get(fertilizer, 0)

        if soil_color_val == 0 or fertilizer_val == 0:
            raise ValueError("Invalid soil color or fertilizer.")

        # Create feature array
        features = np.array([[soil_color_val, nitrogen, phosphorus, potassium, ph, rainfall, temperature, fertilizer_val]])

        # Scale the features
        features_scaled = scaler.transform(features)

        # Predict the crop
        prediction = crop_model.predict(features_scaled)
        predicted_crop = crop_map.get(int(prediction[0]), "Unknown crop")

    except Exception as e:
        logging.error("Error in predict_crop: %s", e)
        return jsonify(error=str(e)), 400

    return jsonify(crop=predicted_crop)

if __name__ == "__main__":
    app.run(debug=True)
