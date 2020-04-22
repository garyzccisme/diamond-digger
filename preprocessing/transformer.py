from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to filter columns, split input dataset into multiple pre-process pipeline.
    """
    def __init__(self, col_name: List):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return X[self.col_name]


class DateSplitTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to break single date column into numerical columns by Year, Month, Day, etc.
    """
    def __init__(self, date_type: str, use_dates: List= None):
        """
        Args:
            date_type: String, must be one of ['First Available Date', 'Last Available Date', 'Delivery Date'].
            use_dates: List, default ['Year', 'Month', 'Day'].
        """
        self._date_type = date_type
        if use_dates is None:
            use_dates = ['Year', 'Month', 'Day']
        self._use_dates = use_dates
        self.split_feature_name = ['{} {}'.format(self._date_type, x) for x in self._use_dates]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        split_date = pd.DataFrame(columns=self.split_feature_name)
        for spec in self._use_dates:
            exec("split_date['{} {}'] = X['{}'].apply(lambda x: x.{})".format(
                self._date_type, spec, self._date_type, spec.lower()))
        return split_date.values


class DateDeltaTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to calculate days difference between two given dates.
    """
    def __init__(self, delta_type: str, former_date: str = None, later_date: str = None):
        """
        Args:
            delta_type: String, must be one of ['deliver_days', 'in_stock_days', 'customized'].
            former_date: if date_type is 'customized',
                then must choose one of ['First Available Date', 'Last Available Date', 'Delivery Date'].
            later_date: if date_type is 'customized',
                then must choose one of ['First Available Date', 'Last Available Date', 'Delivery Date'].
        """
        self.delta_type = delta_type
        self._former_date = former_date
        self._later_date = later_date

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        if self.delta_type == 'deliver_days':
            delta = (X['Delivery Date'] - X['Last Available Date']).apply(lambda x: x.days)
        elif self.delta_type == 'in_stock_days':
            delta = (X['Last Available Date'] - X['First Available Date']).apply(lambda x: x.days)
        elif self.delta_type == 'customized' and self._former_date and self._later_date:
            delta = (X[self._later_date] - X[self._former_date]).apply(lambda x: x.days)
        else:
            raise ValueError("Invalid input")
        # Reshape 1-D array to 2-D array so that can be merged in FeatureUnion() with other features.
        return delta.values.reshape(-1, 1)
