from pandas import DataFrame

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def setup(df: DataFrame, args: dict) -> tuple:
    """Prepare data for model training."""
    # Encode categorial features
    cat_maps = args['categorical_encoding']
    df["neighbourhood"] = df["neighbourhood"].map(cat_maps["MAP_NEIGHB"])
    df["room_type"] = df["room_type"].map(cat_maps["MAP_ROOM_TYPE"])
    # train-test split
    X = df[args['features_selected']].values
    y = df['category'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, **args['train_test_split_args'])
    return X_train, X_test, y_train, y_test


def train(df: DataFrame, args: dict) -> dict:
    """Train model."""
    # Setup
    X_train, X_test, y_train, y_test = setup(df, args)
    # Model
    clf = RandomForestClassifier(**args['model_args'])
    # Training
    clf.fit(X_train, y_train)

    return {"args": args,
            "model": clf,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
            }
