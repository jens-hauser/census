from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference

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

def test_process_data(data):

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
    label = "salary"

    X, y, _, _ = process_data(data,
                        categorical_features=cat_features,
                        label=label, training=True
    )

    assert len(X) > 0
    assert len(y) == len(X)

def test_compute_metrics(model, encoder, label_binarizer, data):
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
    label = "salary"
    X, y, _, _ = process_data(data,
                        categorical_features=cat_features,
                        label=label, training=False, encoder=encoder, lb=label_binarizer
    )
    y_pred = inference(model, X)

    precision, recall, fbeta = compute_model_metrics(y, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

def test_inference(model, data, encoder, label_binarizer):
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
    label = "salary"
    X, _, _, _ = process_data(data,
                        categorical_features=cat_features,
                        label=label, training=False, encoder=encoder, lb=label_binarizer
    )
    y_pred = inference(model, X)
    assert len(y_pred) == len(data)

