'''

Flask API for yield prediction model


'''

from flask import Flask, request, jsonify
import pandas as pd
import joblib

def create_app():
    app = Flask(__name__)

    model = joblib.load('Yield_Prediction.pkl')
    
    with open('original_columns.pkl', 'rb') as f:
        original_columns = joblib.load(f)

    @app.route('/')
    def home():
        return "Welcome to the Yield Prediction System!"

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()

            input_data = pd.DataFrame([data])

            input_data = pd.get_dummies(input_data)

            input_data = input_data.reindex(columns=original_columns, fill_value=0)

            prediction = model.predict(input_data)[0]

            prediction = float(prediction)

            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e)})

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
