import os
from operator import itemgetter

from src.models.main import train_model, evaluate_model
from config.config import DIR_MODELS


def test_train_model():
    train_artifatcs = train_model()
    assert 'simple_classifier.pkl' in os.listdir(DIR_MODELS), \
           'simple_classifier.pkl` was not generated.'
    assert bool(train_artifatcs), 'train artifacts dict was not generated or its empty.'
    assert all(x in ["args", "model", "X_train", "y_train", "X_test", "y_test"] for x in train_artifatcs), \
        "Some train artifacts are missing in the train_model() return dict."


def test_evaluate_model():
    train_artifatcs = train_model()
    args, model, X_train, y_train, X_test, y_test = itemgetter('args',
                                                               'model',
                                                               'X_train',
                                                               'y_train',
                                                               'X_test',
                                                               'y_test')(train_artifatcs)

    evaluation_artifacts = evaluate_model(model, X_test, y_test)
    assert bool(evaluation_artifacts), 'evaluation artifacts dict was not generated or its empty.'