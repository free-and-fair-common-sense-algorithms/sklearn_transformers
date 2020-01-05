from collections import OrderedDict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        if isinstance(x, pd.DataFrame):
            return x[self.columns]
        elif isinstance(x, list):
            return pd.DataFrame(x)[self.columns]
        else:
            return


class MultiColumnLabelEncoder:
    def __init__(self, columns):
        self.columns = columns
        self.encodings = OrderedDict()

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        output = x.copy()
        for col in self.columns:
            encoder = LabelEncoder()
            output[col] = encoder.fit_transform(output[col].astype(str))
            self.encodings[col] = encoder
        return output

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


class BinaryClassifierWithNoise(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier):
        self.classifier = classifier

    def _add_noise(self, predictions):
        adj_predictions = predictions - np.random.rand(len(predictions), 1) * 0.001
        first_value = adj_predictions.take(0, axis=1)
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
