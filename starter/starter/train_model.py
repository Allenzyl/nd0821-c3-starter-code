# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from model import *
from data import *
import pickle
# Add code to load in the data.
data = pd.read_csv("data/census_cleaned.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
data = data.drop(columns = ['Unnamed: 0', 'index'])
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
model = train_model(X_train, y_train)

X_test, y_test, _, _ = process_data(
    test, encoder = encoder, lb = lb, categorical_features=cat_features, label="salary", training=False
)


pickle_out = open("model/classifier.pkl","wb")
pickle.dump(model, pickle_out)
pickle.dump(lb, pickle_out)
pickle_out.close()

pickle_out = open("model/onehot.pkl","wb")
pickle.dump(encoder, pickle_out)
pickle_out.close()

y_pred = inference(model, X_test)
compute_model_metrics(y_test, y_pred)

with open('model/metrics.txt', 'w') as f:
    print('whole model metrics:', compute_model_metrics(y_slice_data, y_slice_pred), file=f)


# slice the data and check model performance
def train_on_slice(cat_var, data, encoder, lb, label):
    for i in data[cat_var].unique():
        slice_data = data[data[cat_var] == i]
        X_slice_data, y_slice_data, _, _ = process_data(
            slice_data, encoder = encoder, lb = lb, categorical_features=cat_features, label=label, training=False
        )
        y_slice_pred = inference(model, X_slice_data)
        with open('starter/model/slice_output.txt', 'a+') as f:
            print('Slice on ' + cat_var + '=' + i + ' metrics: ', compute_model_metrics(y_slice_data, y_slice_pred), file=f)

train_on_slice('education', data, encoder, lb, 'salary')
