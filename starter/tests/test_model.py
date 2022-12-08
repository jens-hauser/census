import pytest
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference

def test_process_data(df, categorical_features, label):
    X, y, _, _ = process_data(df,
                        categorical_features=categorical_features,
                        label=label, training=True
    )
    assert len(X) > 0
    assert len(y) == len(X)

def test_compute_metrics(y, y_pred):
    precision, recall, fbeta = compute_model_metrics(y, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

def test_inference(model, X):
    y_pred = inference(model, X)
    assert len(y_pred) == len(X)

