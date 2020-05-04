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
        self.date_type = date_type
        if use_dates is None:
            use_dates = ['Year', 'Month', 'Day']
        self.use_dates = use_dates
        self.split_feature_name = [self.date_type + ' {}'.format(x) for x in self.use_dates]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        split_date = pd.DataFrame(columns=self.split_feature_name)
        for spec in self.use_dates:
            if spec == "Year":
                split_date[self.date_type + ' {}'.format(spec)] = X[self.date_type].apply(lambda x: x.year)
            elif spec == "Month":
                split_date[self.date_type + ' {}'.format(spec)] = X[self.date_type].apply(lambda x: x.month)
            elif spec == "Day":
                split_date[self.date_type + ' {}'.format(spec)] = X[self.date_type].apply(lambda x: x.day)
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
        self.former_date = former_date
        self.later_date = later_date

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        if self.delta_type == 'deliver_days':
            delta = (X['Delivery Date'] - X['Last Available Date']).apply(lambda x: x.days)
        elif self.delta_type == 'in_stock_days':
            delta = (X['Last Available Date'] - X['First Available Date']).apply(lambda x: x.days)
        elif self.delta_type == 'customized' and self.former_date and self.later_date:
            delta = (X[self.later_date] - X[self.former_date]).apply(lambda x: x.days)
        else:
            raise ValueError("Invalid input")
        # Reshape 1-D array to 2-D array so that can be merged in FeatureUnion() with other features.
        return delta.values.reshape(-1, 1)
