import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin


class RegressorWithNoise(BaseEstimator, RegressorMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def _add_noise(self, predictions):
        return predictions - np.random.random() * 0.001

    def fit(self, x, y):
        return self.regressor.fit(x, y)

    def predict(self, x):
        predictions = self.regressor.predict(x)
        predictions = self._add_noise(predictions)
        return predictions
