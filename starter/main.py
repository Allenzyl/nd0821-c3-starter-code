from fastapi import FastAPI
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import numpy as np
import pandas as pd

# Declare the data object with its components and their type.
class PersonInfo(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
# Define a GET on the specified endpoint.

app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)
process1 = pickle.load(pickle_in)
process2 = pickle.load(pickle_in)

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

@app.post("/predict")
async def predict(info: PersonInfo):
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
    data_in = info.dict()
    X = pd.DataFrame(data_in, index=[0])
    X = X.rename(columns={'marital_status':'marital-status', 'native_country':'native-country'})
    X_categorical = X[cat_features].values

    #print(X_test)
    X_continuous = X.drop(*[cat_features], axis=1)
    X_categorical = process1.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    pred = classifier.predict(X).tolist()[0]
    return {'Salary': pred}


