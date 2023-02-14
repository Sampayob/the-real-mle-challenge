import json

from api.app import app


app.config['DEBUG'] = True
app.config['TESTING'] = True

inputs =  {
            "id": 1001,
            "accommodates": 4,
            "room_type": "Entire home/apt",
            "beds": 2,
            "bedrooms": 1,
            "bathrooms": 2,
            "neighbourhood": "Brooklyn",
            "tv": 1,
            "elevator": 1,
            "internet": 0,
            "latitude": 40.71383,
            "longitude": -73.9658
         }


def test_index_route():
    """Test route ('/')."""
    response = app.test_client().get('/')

    assert response.status_code == 200
    assert response.data.decode('utf-8') == 'NY Estimator API'


def test_request_predict():
    """Test route ('/predict_category')."""
    response=app.test_client().post('/predict_category', 
                        data=json.dumps(inputs),
                        content_type='application/json')

    assert isinstance(response, str), 'response is not a string.'
    assert any(response in category for category in ['Low', 'Mid', 'High', 'Luxury']), \
         'response is not among possible categories.'