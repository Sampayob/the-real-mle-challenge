from pandas import DataFrame

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def setup(df: DataFrame, args: dict) -> tuple:
    """Prepare data for model training."""
    # Encode categorial features
    cat_maps = args['categorical_encoding']
    df["neighbourhood"] = df["neighbourhood"].map(cat_maps["MAP_NEIGHB"])
    df["room_type"] = df["room_type"].map(cat_maps["MAP_ROOM_TYPE"])
    if df["neighbourhood"].isnull().any() or df["neighbourhood"].isnull().any():
        raise ValueError("Some mapping key: value are wrong because 'NaN' values are present.")
    # train-test split
    X = df[args['features_selected']].values
    y = df['category'].values
    if X.ndim != 2:
        raise TypeError("args['features_selected'] is not a feature list.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, **args['train_test_split_args'])
    return X_train, X_test, y_train, y_test


def train(df: DataFrame, args: dict) -> dict:
    """Train model."""
    # Setup
    X_train, X_test, y_train, y_test = setup(df, args)
    if all(x.ndim != 2 for x in [X_train, X_test]):
        ValueError("X_train and/or X_test don't have 2 dimensions.")
    if all(x.ndim != 1 for x in [y_train, y_test]):
        ValueError("y_train and/or y_test don't have 1 dimensions.")
    # Model
    clf = RandomForestClassifier(**args['model_args'])
    # Training
    clf.fit(X_train, y_train)

    return {"args": args,
            "model": clf,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test}
