from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion

from preprocessing.transformer import ColumnSelector, DateDeltaTransformer, DateSplitTransformer


class BaseModel(BaseEstimator):
    def __init__(self, preprocessor_params=None, algo_params=None, cv_params=None):
        self.preprocessor_params = preprocessor_params
        self.algo_params = algo_params
        self.cv_params = cv_params

        self.preprocessor = None
        self.algo = None
        self.pipeline = None

    def build_base_preprocessor(self, inplace=False):
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
