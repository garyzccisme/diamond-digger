import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class DateImputer(BaseEstimator, TransformerMixin):
    """
    Imputer for date type column.
    Default taking all three date columns: 'First Available Date', 'Last Available Date', 'Delivery Date'
    # TODO: add more imputation strategies.
    """
    def __init__(self, first_available_date='earliest', last_available_date='latest', deliver_date='latest'):
        """
        Choose imputation strategy for columns.

        Args:
            first_available_date: Default 'earliest', impute with earliest date.
            last_available_date: Default 'latest', impute with latest date.
            deliver_date: Default 'latest', impute with latest date.
        """
        self.fad = first_available_date
        self.lad = last_available_date
        self.dd = deliver_date

        # self.na_index stores all null index
        self.na_index = None
        self.imputed_values = {}

        # Fit flag to prevent transform before fit
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """
        Generate new dataframe for imputation then store into self.imputed_values.

        Args:
            X: Input data. Only support for pd.DataFrame.

        """
        if self.fad == 'earliest':
            self.imputed_values['First Available Date'] = X['First Available Date'].dropna().min()
        if self.lad == 'latest':
            self.imputed_values['Last Available Date'] = X['Last Available Date'].dropna().max()
        if self.dd == 'latest':
            self.imputed_values['Delivery Date'] = X['Delivery Date'].dropna().max()

        # Change fit flag
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Impute input X's null values with self.imputed_values. Must run after fit().

        Args:
            X: Input data. Only support for pd.DataFrame.

        Returns: Imputed DataFrame

        """
        if not self.is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call 'fit' with appropriate arguments "
                                 "before using this estimator.".format(type(self).__name__))

        X.fillna(value=self.imputed_values, inplace=True)
        return X
