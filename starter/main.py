import os
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.model import inference
from starter.ml.data import process_data


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()

class InputClass(BaseModel):
    age: int
    workclass: str
    fnlgt: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

class PredictionOut(BaseModel):
    pred: int

@app.get("/")
def welcome():
    return {"message": "Welcome!"}

@app.post("/predict", response_model=PredictionOut)
def model_predict(data: InputClass):

    data_dict = data.dict()
    df = pd.DataFrame([data_dict], columns=data_dict.keys())
    df.columns = [col.replace("_", "-") for col in df.columns]

    df.to_csv("data/pred.csv")
    model = joblib.load('model/random_forest_clf.joblib')
    encoder = joblib.load('model/encoder.joblib')
    lb = joblib.load('model/label_binarizer.joblib')
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    X, _, _, _ = process_data(df,
                            categorical_features=cat_features,
                            training=False,
                            encoder=encoder,
                            lb=lb)
    y_pred = model.predict(X)
    return {'pred': y_pred}