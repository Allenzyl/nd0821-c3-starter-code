from model import *
import unittest

def test_train_model(train_model):
    clf = train_model(X_train,y_train)
    assert clf is not None

def test_inference(inference):
    preds = inference(model, X_test)
    assert preds.shape[0] == X_test.shape[0]

def test_compute_model_metrics(compute_model_metrics):
    score1, score2, score3 = compute_model_metrics(y_test, y_pred)
    assert None not in [score1, score2, score3]

if __name__ == '__main__':
    data = pd.read_csv("./starter/data/census_cleaned.csv")
    data = data.drop(columns=['Unnamed: 0', 'index'])
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    test_train_model(train_model)
    model = train_model
    X_test, y_test, _, _ = process_data(
        test, encoder=encoder, lb=lb, categorical_features=cat_features, label="salary", training=False
    )
    y_pred = inference(model, X_test)
    test_inference(inference)
    compute_model_metrics(y_test, y_pred)
    y_pred = inference(model, X_test)
    test_compute_model_metrics(compute_model_metrics)