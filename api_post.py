import requests
import json

data = {
        'age': 60,
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
}
response = requests.post("https://udacity-project-3-census.herokuapp.com/predict/", data=json.dumps(data))
print(response.status_code)
print(response.json())
