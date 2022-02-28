from starter.starter.train_model import *
import unittest
import pandas as pd

def test_train_model(train_model):
    assert train_model is not None

def test_inference(inference):
    assert inference.shape[0] == X_test.shape[0]

def test_compute_model_metrics(compute_model_metrics):
    assert None not in compute_model_metrics

