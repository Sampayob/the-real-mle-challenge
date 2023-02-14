import json


def load_json(filepath: str):
    """Load json file"""
    with open(filepath) as jsonfile:
        return json.load(jsonfile)


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