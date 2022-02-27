#!/usr/bin/env python3.9
from fastapi import FastAPI
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import numpy as np
import pandas as pd
import os
import starter.data

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

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

    class Config:
        schema_extra = {
            "example": {"age": 39,
                        "workclass": " State-gov",
                        "fnlgt": 77516,
                        "education": " Bachelors",
                        "education-num": 13,
                        "marital-status": " Never-married",
                        "occupation": " Adm-clerical",
                        "relationship": " Not-in-family",
                        "race": " White",
                        "sex": " Male",
                        "capital-gain": 2174,
                        "capital-loss": 0,
                        "hours-per-week": 40,
                        "native-country": " United-States"
                    }
        }


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
    data_in = info.dict(by_alias=True)
    X = pd.DataFrame(data_in, index=[0])
    X_categorical = X[cat_features].values

    #print(X_test)
    X_continuous = X.drop(*[cat_features], axis=1)
    X_categorical = process1.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    pred = classifier.predict(X).tolist()[0]
    return {'Salary': pred}


