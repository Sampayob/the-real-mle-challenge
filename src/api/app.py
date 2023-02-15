from flask import Flask, request, jsonify

# from prediction import predict

from src.models.main import predict

def flask_app():
    """Flask app which deploys NY Estimator classification model."""
    app = Flask(__name__)


    @app.route('/')
    def home():
        return "NY Estimator API"


    @app.route('/predict_category', methods=['POST'])
    def start():
        # Get the data from the POST request.
        data = request.get_json()
        # Make prediction
        prediction = predict(data)
        # Build response
        response = {"id": data["id"],
                    "price_category": prediction}
        return jsonify(response)
    
    return app

if __name__ == '__main__':
    app = flask_app()
    app.run(host='0.0.0.0', port=8080, debug=True)
