from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from preprocessing.imputer import DateImputer
from model.base import BaseModel


class DiamondPricer(BaseModel):
    def __init__(self, preprocessor_params=None, algo_params=None, cv=None, cv_params=None):
        super().__init__(preprocessor_params, algo_params, cv, cv_params)
        self.initialization()

    def initialization(self):

        # Initialize preprocessor
        if self.preprocessor is None:
            if self.preprocessor_params is None:
                self.load_base_preprocessor_params()
        self.build_preprocessor()

        # Initialize algorithm
        if self.algo is None:
            if self.algo_params is None:
                self.algo_params = {
                    'algo': 'RandomForestRegressor',
                    'params': {
                        'n_estimators': 100,
                        'criterion': 'mse',
                        'max_depth': None,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': "auto",
                    }
                }
        self.build_algo()

        # Initialize main pipeline
        self.build_pipeline()

        # Initialize cross validation pipeline
        # Currently the cross validation pipeline is only available for algo hyper-parameters tuning
        if self.cv is not None:
            if not self.algo_params.get('tune_params'):
                self.algo_params['tune_params'] = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [2, 4, 3, 5],
                    'max_features': ['sqrt', 'auto'],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [2, 5],
                }
            if self.cv_params is None:
                self.cv_params = {
                    'estimator': self.pipeline,
                    'scoring': None,
                    'cv': None,
                    'refit': True,
                }
            # Wait to add more cv pipeline
            if self.cv == 'GridSearch':
                self.cv_params['param_grid'] = self.algo_params['tune_params']
            elif self.cv == 'RandomizedSearch':
                self.cv_params['param_distributions'] = self.algo_params['tune_params']

            self.build_cv_pipeline()

    def build_preprocessor(self, use_base=True):
        # TODO: add extra feature if use_base=False
        if use_base:
            self.build_base_preprocessor(inplace=True)

    def build_algo(self):
        # TODO: add more choices for algorithm
        if self.algo_params['algo'] == 'RandomForestRegressor':
            self.algo = RandomForestRegressor(**self.algo_params['params'])

    def build_pipeline(self, prefix=None):
        if prefix is None:
            prefix = ['preprocessor', 'algo']
        self.pipeline = Pipeline([(prefix[0], self.preprocessor), (prefix[1], self.algo)])

    def build_cv_pipeline(self):
        if self.cv == 'GridSearch':
            self.cv_pipeline = GridSearchCV(**self.cv_params)
        elif self.cv == 'RandomizedSearch':
            self.cv_pipeline = RandomizedSearchCV(**self.cv_params)
