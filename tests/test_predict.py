from src.models.main import predict


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


def test_predict():
    result = predict(inputs)

    assert isinstance(result, str), 'Predicted value/s is not a categorical value.'