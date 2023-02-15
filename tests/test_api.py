import json

import pytest

from src.api.app import flask_app


@pytest.fixture
def app():
    yield flask_app()


def test_index_route(app):
    response = app.test_client().get('/')

    assert response.status_code == 200
    assert response.data.decode('utf-8') == 'NY Estimator API'


def test_request_predict(app):
    inputs =  {"id": 1001,
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
               "longitude": -73.9658}

    response=app.test_client().post('/predict_category', 
                        data=json.dumps(inputs),
                        content_type='application/json')

    response = json.loads(response.data)

    assert isinstance(response, dict), 'response is not a dict.'
    assert any(response['price_category'] in category for category in ['Low', 'Mid', 'High', 'Luxury']), \
         'response is not among possible categories.'