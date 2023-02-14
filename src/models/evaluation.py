import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import (roc_auc_score,
                             classification_report,
                             confusion_matrix)
from sklearn.base import BaseEstimator


def ovr_roc_auc_score(clf: BaseEstimator,
                      X_test: DataFrame,
                      y_test: Series,
                      ) -> float:
    """Compute overall one-versus-rest area under the ROC."""
    y_proba = clf.predict_proba(X_test)
    return roc_auc_score(y_test, y_proba, multi_class='ovr')


def plot_feature_importance(clf: BaseEstimator, X_train: DataFrame) -> None:
    """Plot feature importance bar plot."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_train.columns[indices]
    importances = importances[indices]

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.barh(range(len(importances)), importances)
    plt.yticks(range(len(importances)), features, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance", fontsize=12)

    plt.show()


def plot_confusion_matrix(y_test: Series, y_pred: Series) -> None:
    """Plot confussion matrix."""
    classes = [0, 1, 2, 3]
    labels = ['low', 'mid', 'high', 'lux']

    cm = confusion_matrix(y_test, y_pred)
    cm = cm / cm.sum(axis=1).reshape(len(classes), 1)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm,
                annot=True,
                cmap='BuGn',
                square=True,
                fmt='.2f',
                annot_kws={'size': 10},
                cbar=False)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Real', fontsize=16)
    plt.xticks(ticks=np.arange(.5, len(classes)), labels=labels, rotation=0, fontsize=12)
    plt.yticks(ticks=np.arange(.5, len(classes)), labels=labels, rotation=0, fontsize=12)
    plt.title("Simple model", fontsize=18)

    plt.show()


def custom_classificaton_report(y_test: Series, y_pred: Series) -> DataFrame:
    """Create report DataFrame from report dict."""
    maps = {'0.0': 'low', '1.0': 'mid', '2.0': 'high', '3.0': 'lux'}

    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame.from_dict(report).T[:-3]
    df_report.index = [maps[i] for i in df_report.index]
    return df_report


def plot_metrics(df_report: DataFrame) -> None:
    """Plot model metrics as multiple bar plots."""
    metrics = ['precision', 'recall', 'support']
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 7))

    for i, ax in enumerate(axes):

        ax.barh(df_report.index, df_report[metrics[i]], alpha=0.9)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel(metrics[i], fontsize=12)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("Simple model", fontsize=14)
    plt.show()