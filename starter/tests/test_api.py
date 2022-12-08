def test_welcome(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome!"}


def test_predict_pos(client):
    request = client.post("/predict", json={'age': 45,
                                     'workclass': 'Private',
                                     'fnlgt': 164190,
                                     'education': 'Masters',
                                     'education-num': 14,
                                     'marital_status': 'Never-married',
                                     'occupation': 'Prof-specialty',
                                     'relationship': 'Not-in-family',
                                     'race': 'White',
                                     'sex': 'Male',
                                     'capital-gain': 0,
                                     'capital-loss': 0,
                                     'hoursPerWeek': 60,
                                     'nativeCountry': 'United-States'
                                     })
    assert request.status_code == 200
    assert request.json() == {"prediction": ">50K"}


def test_predict_low(client):
    request = client.post("/predict", json={'age': 21,
                                     'workclass': 'Private',
                                     'fnlgt': 164190,
                                     'education': 'HS-grad',
                                     'education-num': 9,
                                     'marital_status': 'Never-married',
                                     'occupation': 'Prof-specialty',
                                     'relationship': 'Not-in-family',
                                     'race': 'Black',
                                     'sex': 'Female',
                                     'capital-gain': 0,
                                     'capital-loss': 0,
                                     'hoursPerWeek': 40,
                                     'nativeCountry': 'United-States'
                                     })
    assert request.status_code == 200
    assert request.json() == {"prediction": "<=50K"}