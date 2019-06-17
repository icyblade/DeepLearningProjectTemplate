"""Sample main module."""
import numpy as np


class MeanRegressionModel:
    def fit(self, features, labels=None):  # pylint: disable=unused-argument
        return self

    def predict(self, features):
        return np.mean(features, axis=1)
