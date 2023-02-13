import pickle

from sklearn.metrics import accuracy_score
import pandas as pd
from numpy import array

from config.config import (FILEPATH_MODEL,
                           FILEPATH_DATA_PROCESSED,
                           FILEPATH_ARGS,
                          ) 
from src.models.train import train
from src.models.predict import predict_pipeline
from src.models.evaluation import (ovr_roc_auc_score,
                                   custom_classificaton_report,
                                  )
from src.utils.utils import load_json


def train_model():
    """Train model on processed data."""
    # Load labeled data
    df = pd.read_csv(str(FILEPATH_DATA_PROCESSED), index_col=0)
    # Drop null values
    df = df.dropna(axis=0)
    # Train
    args = load_json(str(FILEPATH_ARGS))
    train_artifatcs = train(df, args)
    model = train_artifatcs['model']
    pickle.dump(model, open(FILEPATH_MODEL, 'wb'))
    return train_artifatcs


def evaluate_model(model, X_test, y_test):
    """Evaluate trained model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc_score = ovr_roc_auc_score(model, X_test, y_test)
    classf_report = custom_classificaton_report(y_test, y_pred)

    print('\n')
    print('Accuracy: ', accuracy)
    print('ROC AUC Score (OVR): ', roc_auc_score)
    print('\n')
    print(classf_report)

    return  {"accuracy": accuracy,
            "ovr_roc_auc_score": roc_auc_score,
            "classficiation_report": classf_report,
            }


def predict(inputs: dict) -> array:
    """Make a prediction from trained model."""
    args = load_json(str(FILEPATH_ARGS))
    result = predict_pipeline(inputs, args)
    decoded_result = args['label_encoding'][str(int(result))]
    return decoded_result
