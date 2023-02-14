import json
import pickle
import numpy as np
from numpy import array


def encode_categorical_features(inputs: dict, cat_encodings: dict) -> dict:
    """Encode categorical features present in the input data before inference."""
     # Find values to encode
    neighb_toencode = [(k, v) for k, v in cat_encodings["MAP_NEIGHB"].items() if k in inputs.values()][0]
    room_type_toencode = [(k, v) for k, v in cat_encodings["MAP_ROOM_TYPE"].items() if k in inputs.values()][0]
    # Replace categorical with integer value
    if len(neighb_toencode) > 0:
        inputs['neighbourhood'] = neighb_toencode[1]
    if len(room_type_toencode) > 0:
        inputs['room_type'] = room_type_toencode[1]
    
    return inputs


def predict_pipeline(inputs: dict, train_args: dict) -> int:
    """Prepocess input data and feed it to the model to obtain predictions."""
    # Encode categorial features
    inputs = encode_categorical_features(inputs, train_args['categorical_encoding'])
    # Select features
    inputs = {k:v for k, v in inputs.items() if k in train_args['features_selected']}
    # Predict
    inputs = np.array([[v for k, v in inputs.items()]])
    if inputs.ndim != 2:
        ValueError("'inputs' array does not have 2 dimensions.")
    with open('simple_classifier.pkl', 'rb') as file:
        model = pickle.load(file)
    y_pred = model.predict(inputs)
    return y_pred


def predict(inputs: dict) -> array:
    """Make a prediction from trained model."""
    with open('args.json') as jsonfile:
        args = json.load(jsonfile)
    result = predict_pipeline(inputs, args)
    if 'label_encoding' in args:
            decoded_result = args['label_encoding'][str(int(result))]
            return decoded_result
    else:
        raise ValueError("'label encoding' not in args.json file.")
