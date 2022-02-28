import requests

def test_api_live_predict_1():
    r = requests.post('https://salarypredicter.herokuapp.com/predict/',
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
    print(r.status_code,
          r.json())

if __name__ == "__main__":
    test_api_live_predict_1()