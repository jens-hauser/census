import pytest
import os
import sys
import pandas as pd
import joblib

from fastapi.testclient import TestClient

#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from main import app


@pytest.fixture(scope='session')
def data():
    """
        Get dataset
    """
    root_path = os.path.dirname(os.path.abspath(__file__))
    file_name = "census_clean.csv"
    file_path = os.path.join(root_path, "data", file_name)

    df = pd.read_csv(file_path, low_memory=False)
    return df


@pytest.fixture(scope='session')
def model():
    """
        Get Model
    Returns
    -------
    """
    root_path = os.path.dirname(os.path.abspath(__file__))
    model_name = "random_forest_clf.joblib"
    trained_model = joblib.load(os.path.join(root_path, "model", model_name))

    return trained_model

@pytest.fixture(scope='session')
def encoder():
    """
        Get Model
    Returns
    -------
    """
    root_path = os.path.dirname(os.path.abspath(__file__))
    encoder_name = "encoder.joblib"
    encoder = joblib.load(os.path.join(root_path, "model", encoder_name))

    return encoder

@pytest.fixture(scope='session')
def label_binarizer():
    """
        Get Model
    Returns
    -------
    """
    root_path = os.path.dirname(os.path.abspath(__file__))
    name = "label_binarizer.joblib"
    label_binarizer = joblib.load(os.path.join(root_path, "model", name))

    return label_binarizer

@pytest.fixture(scope='session')
def cat_features():
    """
        Get Model
    Returns
    -------
    """
    return [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]


@pytest.fixture(scope='session')
def client():
    """
    Get dataset
    """
    client = TestClient(app)
    return client