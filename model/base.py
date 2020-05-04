from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from preprocessing.imputer import DateImputer
from preprocessing.transformer import ColumnSelector, DateDeltaTransformer, DateSplitTransformer


class BaseModel(BaseEstimator):
    """
    Base Model for all customized model pipelines.
    # TODO: base attributes & methods need to be more considered.
    """
    def __init__(self, preprocessor_params: Dict = None, algo_params: Dict = None,
                 cv: str = None, cv_params: Dict = None):
        """
        Args:
            preprocessor_params: Dict, stores all hyper-parameters for pre-processing pipeline.
            algo_params: Dict, stores all hyper-parameters for algorithm estimator.
            cv: String, can be one of ['GridSearch', 'RandomizedSearch'].
            cv_params: Dict, stores all hyper-parameters for cross validation process.
        """
        self.preprocessor_params = preprocessor_params
        self.algo_params = algo_params
        self.cv = cv
        self.cv_params = cv_params

        self.preprocessor = None
        self.feature_name = []
        self.algo = None
        self.pipeline = None
        self.cv_pipeline = None

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
        self.feature_name += self.preprocessor_params['cat_col']

        # Numerical Features
        num_preprocessor = Pipeline([
            ('selector', ColumnSelector(self.preprocessor_params['num_col'])),
            ('imputer', self.preprocessor_params['num_imputer']),
            ('scaler', self.preprocessor_params['num_scaler'])
        ])
        # Add numerical features name
        self.feature_name += self.preprocessor_params['num_col']

        # Datetime Features
        # Make datetime splitter
        splitter = FeatureUnion(transformer_list=[
            ('{}'.format(x), DateSplitTransformer(x)) for x in self.preprocessor_params['date_splitter']
        ])
        self.feature_name += [name for x in splitter.transformer_list for name in x[1].split_feature_name]
        # Make datetime delta
        delta = FeatureUnion(transformer_list=[
            ('{}'.format(x), DateDeltaTransformer(x)) for x in self.preprocessor_params['date_delta']
        ])
        self.feature_name += self.preprocessor_params['date_delta']

        date_feature = FeatureUnion(transformer_list=[
            ('splitter', splitter),
            ('delta', delta),
        ])
        date_preprocessor = Pipeline([
            ('selector', ColumnSelector(self.preprocessor_params['date_col'])),
            ('imputer', self.preprocessor_params['date_imputer']),
            ('date_feature', date_feature)
        ])

        # Make total feature union
        base_preprocessor = FeatureUnion(transformer_list=[
            ('cat_preprocessor', cat_preprocessor),
            ('num_preprocessor', num_preprocessor),
            ('date_preprocessor', date_preprocessor),
        ])

        # Unify self.feature_name
        self.feature_name = [name.lower().replace(' ', '_') for name in self.feature_name]

        if inplace:
            self.preprocessor = base_preprocessor
        else:
            return base_preprocessor

    def load_base_preprocessor_params(self, cat_encoder):
        """
        Load base preprocessor_params, which would be share for all models.
        Args:
            cat_encoder: CategoricalEncoder() in sklearn.preprocessing or self designed.

        """
        self.preprocessor_params = {
            'target_col': ['Price'],
            'cat_col': ['Shape', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Fluorescence', 'Culet'],
            'cat_encoder': cat_encoder,
            'cat_imputer': SimpleImputer(strategy='most_frequent'),
            'num_col': ['Carat', 'Depth', 'Table', 'L/W'],
            'num_imputer': SimpleImputer(strategy="median"),
            'num_scaler': StandardScaler(),
            'date_col': ['First Available Date', 'Last Available Date', 'Delivery Date'],
            'date_imputer': DateImputer(),
            'date_splitter': ['First Available Date'],
            'date_delta': ['deliver_days', 'in_stock_days'],
        }

    def build_preprocessor(self):
        return

    def build_algo(self):
        return

    def build_pipeline(self):
        return

    def build_cv_pipeline(self):
        return

    def fit(self, X, y):
        return

    def cv_fit(self, X, y):
        return

    def predict(self, X):
        return
