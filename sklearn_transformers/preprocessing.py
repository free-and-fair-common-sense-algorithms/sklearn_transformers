from collections import OrderedDict
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
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
