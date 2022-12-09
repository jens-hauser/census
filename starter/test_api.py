from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_welcome():
    response = client.get("/")
    assert response.status_code == 200, response.json()
    assert response.json() == {"message": "Welcome!"}


def test_predict_pos():
    response = client.post("/predict", json={'age': 60,
                                     'workclass': 'Private',
                                     'fnlgt': 45781,
                                     'education': 'Masters',
                                     'education_num': 14,
                                     'marital_status': 'Never-married',
                                     'occupation': 'Prof-specialty',
                                     'relationship': 'Not-in-family',
                                     'race': 'White',
                                     'sex': 'Male',
                                     'capital_gain': 14000,
                                     'capital_loss': 0,
                                     'hours_per_week': 60,
                                     'native_country': 'United-States'
                                     })
    assert response.status_code == 200, response.json()
    assert response.json() == {"pred": 1}


def test_predict_low():
    response = client.post("/predict", json={'age': 21,
                                     'workclass': 'Private',
                                     'fnlgt': 164190,
                                     'education': 'HS-grad',
                                     'education_num': 9,
                                     'marital_status': 'Never-married',
                                     'occupation': 'Prof-specialty',
                                     'relationship': 'Not-in-family',
                                     'race': 'Black',
                                     'sex': 'Female',
                                     'capital_gain': 0,
                                     'capital_loss': 0,
                                     'hours_per_week': 40,
                                     'native_country': 'United-States'
                                     })
    assert response.status_code == 200, response.json()
    assert response.json() == {"pred": 0}