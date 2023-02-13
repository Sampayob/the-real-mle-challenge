import pickle

from config.config import FILEPATH_MODEL
from src.utils.utils import encode_categorical_features


def predict_pipeline(inputs: dict, train_args: dict) -> int:
    """Prepocess input data and feed it to the model to obtain predictions."""
    # Encode categorial features
    inputs = encode_categorical_features(inputs, train_args['categorical_encoding'])
    # Select features
    inputs = {k:v for k, v in inputs.items() if k in train_args['features_selected']}
    # Predict
    inputs = [[v for k, v in inputs.items()]] 
    model = pickle.load(open(str(FILEPATH_MODEL), 'rb'))
    y_pred = model.predict(inputs)
    return y_pred
