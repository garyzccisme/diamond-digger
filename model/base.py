from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion

from preprocessing.transformer import ColumnSelector, DateDeltaTransformer, DateSplitTransformer


class BaseModel(BaseEstimator):
    """
    Base Model for all customized model pipelines.
    # TODO: base attributes & methods need to be more considered.
    """
    def __init__(self, preprocessor_params: Dict = None, algo_params: Dict = None, cv_params: Dict = None):
        """
        Args:
            preprocessor_params: Dict, stores all hyper-parameters for pre-processing pipeline.
            algo_params: Dict, stores all hyper-parameters for algorithm estimator.
            cv_params: Dict, stores all hyper-parameters for cross validation process.
        """
        self.preprocessor_params = preprocessor_params
        self.algo_params = algo_params
        self.cv_params = cv_params

        self.preprocessor = None
        self.algo = None
        self.pipeline = None

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
        cat_preprocessor = Pipeline([
            ('selector', ColumnSelector(self.preprocessor_params['cat_col'])),
            ('imputer', self.preprocessor_params['cat_imputer']),
            ('encoder', self.preprocessor_params['cat_encoder']),
        ])

        # Numerical Features
        num_preprocessor = Pipeline([
            ('selector', ColumnSelector(self.preprocessor_params['num_col'])),
            ('imputer', self.preprocessor_params['num_imputer']),
            ('scaler', self.preprocessor_params['num_scaler'])
        ])

        # Datetime Features
        splitter = FeatureUnion(transformer_list=[
            ('{}'.format(x), DateSplitTransformer(x)) for x in self.preprocessor_params['date_splitter']
        ])
        delta = FeatureUnion(transformer_list=[
            ('{}'.format(x), DateDeltaTransformer(x)) for x in self.preprocessor_params['date_delta']
        ])
        date_feature = FeatureUnion(transformer_list=[
            ('splitter', splitter),
            ('delta', delta),
        ])
        date_preprocessor = Pipeline([
            ('selector', ColumnSelector(self.preprocessor_params['date_col'])),
            ('imputer', self.preprocessor_params['date_imputer']),
            ('date_feature', date_feature)
        ])

        # Make Union
        base_preprocessor = FeatureUnion(transformer_list=[
            ('cat_preprocessor', cat_preprocessor),
            ('num_preprocessor', num_preprocessor),
            ('date_preprocessor', date_preprocessor),
        ])

        if inplace:
            self.preprocessor = base_preprocessor
        else:
            return base_preprocessor

    def build_preprocessor(self):
        return

    def load_algo(self):
        return

    def build_pipeline(self):
        return
