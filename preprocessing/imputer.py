import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class DateImputer(BaseEstimator, TransformerMixin):
    def __init__(self, first_available_date='earliest', last_available_date='latest', deliver_date='latest'):
        self.fad = first_available_date
        self.lad = last_available_date
        self.dd = deliver_date
        self.na_index = None
        self.imputed_values = {}
        self.is_fitted = False

    def _validate_input(self, X):
        self.na_index = {
            'fad': X[X['First Available Date'].isna()].index,
            'lad': X[X['Last Available Date'].isna()].index,
            'dd': X[X['Delivery Date'].isna()].index,
        }

    def fit(self, X, y=None):
        self._validate_input(X)
        if self.fad == 'earliest':
            self.imputed_values['fad'] = X.loc[self.na_index['fad'], 'First Available Date'].apply(
                lambda x: X['First Available Date'].dropna().min()
            )
        if self.lad == 'latest':
            self.imputed_values['lad'] = X.loc[self.na_index['lad'], 'Last Available Date'].apply(
                lambda x: X['Last Available Date'].dropna().max()
            )
        if self.dd == 'latest':
            self.imputed_values['dd'] = X.loc[self.na_index['dd'], 'Delivery Date'].apply(
                lambda x: X['Delivery Date'].dropna().max()
            )
        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        if not self.is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call 'fit' with appropriate arguments "
                                 "before using this estimator.".format(type(self).__name__))

        X.loc[self.na_index['fad'], 'First Available Date'] = self.imputed_values['fad']
        X.loc[self.na_index['lad'], 'Last Available Date'] = self.imputed_values['lad']
        X.loc[self.na_index['dd'], 'Delivery Date'] = self.imputed_values['dd']
        return X
