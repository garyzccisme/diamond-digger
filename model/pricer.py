from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV

from preprocessing.imputer import DateImputer
from model.base import BaseModel


class DiamondPricer(BaseModel):
    def __init__(self, preprocessor_params=None, algo_params=None, is_cv=False, cv_params=None):
        super().__init__(preprocessor_params, algo_params, is_cv, cv_params)
        self.initialization()

    def initialization(self):

        # Initialize preprocessor
        if self.preprocessor is None:
            if self.preprocessor_params is None:
                self.load_base_preprocessor_params(cat_encoder=OrdinalEncoder())
            self.build_preprocessor()

        # Initialize algorithm
        if self.algo is None:
            if self.algo_params is None:
                self.algo_params = {
                    'algo': 'RandomForestRegressor',
                    'hyper_params': {
                        'n_estimators': 100,
                        'criterion': 'mse',
                        'max_depth': None,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'min_weight_fraction_leaf': 0.,
                        'max_features': "auto",
                        'max_leaf_nodes': None,
                        'min_impurity_decrease': 0.,
                        'min_impurity_split': None,
                        'bootstrap': True,
                        'oob_score': False,
                        'n_jobs': None,
                        'random_state': None,
                        'verbose': 0,
                        'warm_start': False,
                        'ccp_alpha': 0.0,
                        'max_samples': None
                    }
                }
            self.build_algo()

        if self.is_cv:
            if self.cv_params is None:
                self.cv_params = {
                    'cv': 'RandomizedSearchCV',
                    # TODO: need function to design the name of features, need to refactor self.build_preprocessor()
                    # https://github.com/garyzccisme/SimpleBet/blob/master/researchlab/researchlab/pipelines/utils.py
                    'cv_hyper_params': {},
                }

    def build_preprocessor(self, use_base=True):

        if use_base:
            self.build_base_preprocessor(inplace=True)
        # TODO: add extra feature if use_base=False

    def build_algo(self):

        if self.algo_params['algo'] == 'RandomForestRegressor':
            self.algo = RandomForestRegressor(**self.algo_params['hyper_params'])
        # TODO: add more choices for algorithm

    def build_pipeline(self):

        self.pipeline = Pipeline([('preprocessor', self.preprocessor), ('algo', self.algo)])

    def build_cv_pipeline(self):

        if self.cv_params['cv'] == 'RandomizedSearchCV':
            self.cv_pipeline = RandomizedSearchCV(**self.cv_params['cv_hyper_params'])
        # TODO: add more choices for cv









