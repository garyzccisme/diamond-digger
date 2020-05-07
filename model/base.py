from typing import Iterable, Dict, Optional

from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from preprocessing.imputer import DateImputer
from preprocessing.transformer import ColumnSelector, DateDeltaTransformer, DateSplitTransformer
from preprocessing.utils import generate_cat_preprocessor, generate_date_preprocessor, generate_feature_union, \
    generate_num_preprocessor


class BaseModel(BaseEstimator):
    """
    Base Model for all customized model pipelines.

    """
    def __init__(self, preprocessor_params: Dict = None, algo_params: Dict = None,
                 cv: str = None, cv_params: Dict = None):
        """
        Args:
            preprocessor_params: Dict, stores all hyper-parameters for pre-processing pipeline.
                `BaseModel.load_base_preprocessor_params()` gives out a basic format.
            algo_params: Dict, stores all hyper-parameters for algorithm estimator.
                Format: {
                    'algo': algorithm enum (String),
                    'params': algorithm parameters (Dict),
                    'tune_params' (optional): algorithm parameter distribution,
                    }
            cv: String, should be one of ['GridSearch', 'RandomizedSearch'].
            cv_params: Dict, stores all hyper-parameters for cross validation process.
                Note that`tuning params distribution` names differently among CV-Pipelines.
                It can also be given in `algo_params[tune_params]`, which has priority to overwrite the one in
                `cv_params` if conflicts. Thus it would better to define in `algo_params`.
        """
        self.preprocessor_params = preprocessor_params
        self.algo_params = algo_params
        self.cv = cv
        self.cv_params = cv_params

        self.preprocessor = None
        self.preprocessor_tuning_params = None
        self.feature_name = None
        self.algo = None
        self.pipeline = None
        self.cv_pipeline = None

        self.prediction = None
        self.metrics = {}

    def initialization(self):
        """
        This method is supposed to initialize each component for the model.
        """
        raise NotImplementedError('Need to overwrite in subclass')

    def load_base_preprocessor_params(self):
        """
        Load base preprocessor_params, which would be share for all models.

        """
        self.preprocessor_params = {
            'cat': {
                'columns': ['Shape', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Fluorescence', 'Culet'],
                'imputer_strategy': 'most_frequent',
                'encoder_type': 'Ordinal',
                'tune_params': None,
            },
            'num': {
                'columns': ['Carat', 'Depth', 'Table', 'L/W'],
                'imputer_strategy': 'median',
                'scaler_type': 'Standard',
                'tune_params': None,
            },
            'date': {
                'split_cols': ['First Available Date'],
                'delta_types': ['deliver_days', 'in_stock_days'],
                'imputer_strategy': None,
            },
        }

    def build_base_preprocessor(self, inplace: bool = False):
        """
        Build basic features for all models, other customized features can be added by `build_preprocessor()`.
        Basic features include Categorical, Numerical, Datetime three main types.
        The specific columns are determined by self.preprocessor_params.

        Args:
            inplace: bool, if true then update self.preprocessor, if false then return preprocesser.

        Returns:

        """
        # Categorical Features
        cat_preprocessor, cat_feature_name, cat_tuning_dict = generate_cat_preprocessor(
            **self.preprocessor_params['cat']
        )

        # Numerical Features
        num_preprocessor, num_feature_name, num_tuning_dict = generate_num_preprocessor(
            **self.preprocessor_params['num']
        )

        # Datetime Features
        date_preprocessor, date_feature_name = generate_date_preprocessor(**self.preprocessor_params['date'])

        # Make total FeatureUnion
        transformer_dict_list = [
            {'prefix': 'CAT', 'transformer': cat_preprocessor, 'tuning_params': cat_tuning_dict},
            {'prefix': 'NUM', 'transformer': num_preprocessor, 'tuning_params': num_tuning_dict},
            {'prefix': 'DATE', 'transformer': date_preprocessor},
        ]
        base_preprocessor, self.preprocessor_tuning_params = generate_feature_union(transformer_dict_list)

        # Unify self.feature_name
        self.feature_name = cat_feature_name + num_feature_name + date_feature_name
        self.feature_name = [name.lower().replace(' ', '_') for name in self.feature_name]

        if inplace:
            self.preprocessor = base_preprocessor
        else:
            return base_preprocessor

    def fit(self, X, y, tune=False):
        """
        Train model, which is to fit self.pipeline.
        Args:
            X: iterable, Training data.
            y: iterable, Training target.
            tune: bool, if True then run cv_fit first and replace self.pipeline with tuned one.

        Returns: self, this model.

        """
        if tune:
            self.cv_fit(X, y)
        else:
            self.pipeline.fit(X, y)
        return self

    def cv_fit(self, X, y, replace=True):
        """
        Hyper-parameters tuning for self.pipeline, which is to fit self.cv_pipeline.
        Args:
            X: iterable, Training data.
            y: iterable, Training target.
            replace: bool, if True then replace self.pipeline with tuned one.

        """
        self.cv_pipeline.fit(X, y)
        if replace:
            self.pipeline = self.cv_pipeline.best_estimator_

    def predict(self, X):
        """
        Predict with trained model.
        Args:
            X: iterable, Testing data.

        Returns: predicted y, array-like.

        """
        self.prediction = self.pipeline.predict(X)
        return self.prediction

    def score(self, X, y, metrics: Optional[Iterable] = None):
        """
        Get Scores(Metrics) for prediction.
        Args:
            X: iterable, Testing data.
            y: iterable, Training target.
            metrics: iterable, List of metric names. If None then return self.pipeline default score.

        Returns: float or Dict

        """
        if self.prediction is None or self.prediction.shape != y.shape:
            self.predict(X)

        if metrics is None:
            return self.pipeline.score(X, y)

        for metric in metrics:
            if metric == 'mse':
                self.metrics['mse'] = mean_squared_error(y_true=y, y_pred=self.prediction)
            elif metric == 'mae':
                self.metrics['mae'] = mean_absolute_error(y_true=y, y_pred=self.prediction)
            elif metric == 'r-square':
                self.metrics['r-square'] = r2_score(y_true=y, y_pred=self.prediction)
        return self.metrics

    def build_preprocessor(self):
        raise NotImplementedError('Need to overwrite in subclass')

    def build_algo(self):
        raise NotImplementedError('Need to overwrite in subclass')

    def build_pipeline(self):
        raise NotImplementedError('Need to overwrite in subclass')

    def build_cv_pipeline(self):
        raise NotImplementedError('Need to overwrite in subclass')


