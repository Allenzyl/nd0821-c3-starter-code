from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_api_locally_predict_0():
    r = client.post("/predict",
                    json={
                        "age": 39,
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
                    })
    assert r.status_code == 200
    assert r.json() == {"Salary": 0}


def test_api_locally_predict_1():
    r = client.post("/predict",
                    json={
                          "age": 37,
                          "workclass": " Private",
                          "fnlgt": 280464,
                          "education": " Some-college",
                          "education-num": 10,
                          "marital-status": " Married-civ-spouse",
                          "occupation": " Exec-managerial",
                          "relationship": " Husband",
                          "race": " Black",
                          "sex": " Male",
                          "capital-gain": 0,
                          "capital-loss": 0,
                          "hours-per-week": 80,
                          "native-country": " United-States"
                        })
    assert r.status_code == 200
    assert r.json() == {"Salary": 1}