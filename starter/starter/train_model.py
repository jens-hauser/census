# Script to train machine learning model.
import os
import json
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, performance_on_slices
# Add code to load in the data.

data = pd.read_csv("starter/data/census_clean.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",  training=False, encoder=encoder, lb=lb
)
# Train and save a model.
clf = train_model(X_train, y_train)
dump(clf, 'starter/model/random_forest_clf.joblib') 
dump(encoder, 'starter/model/encoder.joblib') 
dump(lb, 'starter/model/label_binarizer.joblib') 

# Check performance on test set
y_pred = inference(clf, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print ("Precision: ", precision) 
print ("Recall: ", recall)
print ("fbeta: ", fbeta)

# Check performance on test set
performance = performance_on_slices(clf, test, categorical_features=cat_features, encoder=encoder, lb=lb, label="salary")
#json_string = json.dumps(performance, indent=4, sort_keys=True)
#print(json_string)

with open('starter/data/slice_output.txt', 'w') as outfile:
    json.dump(str(performance), outfile)


