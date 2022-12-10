from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = RandomForestClassifier(n_estimators=200, max_depth=9, max_features="sqrt")
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def performance_on_slices(model, data, categorical_features, encoder, lb, label="salary"):
    """
    Computes performance metrics on slices of the data for categorical features
    """
    slices = {}
    for feature in categorical_features:
        for value in data[feature].unique():
            X_slice, y_slice, _, _ = process_data(data[data[feature] == value], categorical_features, label=label, training=False, encoder=encoder, lb=lb)
            y_pred = inference(model, X_slice)
            slices[feature + '_' + value] = compute_model_metrics(y_slice, y_pred)

    return slices

def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
