import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class BinaryClassifierWithNoise(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier):
        self.classifier = classifier

    def _add_noise(self, predictions):
        adj_predictions = predictions - np.random.rand(len(predictions), 1) * 0.001
        first_value = np.absolute(adj_predictions).take(0, axis=1)
        second_value = 1 - first_value
        return np.stack([first_value, second_value], axis=1)

    def _add_target_labels(self, predictions):
        return [dict(zip(self.classifier.classes_, prediction)) for prediction in predictions]

    def fit(self, x, y):
        return self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict(x)

    def predict_proba(self, x):
        predictions = self.classifier.predict_proba(x)
        predictions = self._add_noise(predictions)
        return self._add_target_labels(predictions)
