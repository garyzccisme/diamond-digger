import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self._col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._col_name]


class DateSplitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_type, use_dates=None):
        self._date_type = date_type
        if use_dates is None:
            use_dates = ['Year', 'Month', 'Day']
        self._use_dates = use_dates

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        split_date = pd.DataFrame(columns=['{} {}'.format(self._date_type, x) for x in self._use_dates])
        for spec in self._use_dates:
            exec("split_date['{} {}'] = X['{} Date'].apply(lambda x: x.{})".format(
                self._date_type, spec, self._date_type, spec.lower()))
        return split_date.values


class DateDeltaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, delta_type, former_date=None, later_date=None):
        self.delta_type = delta_type
        self._former_date = former_date
        self._later_date = later_date

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.delta_type == 'deliver_days':
            delta = (X['Delivery Date'] - X['Last Available Date']).apply(lambda x: x.days)
        elif self.delta_type == 'in_stock_days':
            delta = (X['First Available Date'] - X['Last Available Date']).apply(lambda x: x.days)
        elif self.delta_type == 'customized' and self._former_date and self._later_date:
            delta = (X[self._later_date] - X[self._former_date]).apply(lambda x: x.days)
        else:
            raise ValueError("Invalid input")
        return delta.values
