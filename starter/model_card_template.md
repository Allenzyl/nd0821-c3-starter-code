# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Yilun created this model. It is random forest using default hyperparameters in sklearn 0.23.2.

## Intended Use
The model should be used to predict the salary based on a person's information such as age, education, etc.

## Training Data
The data was obtained from https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
The cleaned dataset has shape of 32537 rows * 15 columns, and a 80-20 split was used to break this into a train and test data set.

## Metrics
Fbeta, precision and recall are used as the model performance, and they are respectively: 
(0.9853658536585366, 0.8278688524590164, 0.8997772828507795)

## Ethical Considerations
The model should not be used for privacy fields.

## Caveats and Recommendations
Please note the model is NOT cross-validated and may be overfitted.