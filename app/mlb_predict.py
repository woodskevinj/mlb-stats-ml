from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('models/mlb_salary_model.joblib')

@app.route("/")
def home():
    return jsonify({"message": "MLB OBP Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Create DataFram from input
    df = pd.DataFrame([data])

    # Predict OBP
    predicted_obp = model.predict(df)[0]

    return jsonify({
        "predicted_OBP": round(float(predicted_obp), 3)
    })

if __name__ == "__main__":
    app.run(debug=True)