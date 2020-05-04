from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

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

        # Initialize main pipeline
        self.build_pipeline()

        # Initialize cross validation pipeline
        # Currently the cross validation pipeline is only available for algo hyper-parameters tuning
        if self.cv is not None:
            if not self.algo_params.get('hyper_params_distribution'):
                self.algo_params['hyper_params_distribution'] = {
                                                                    'algo__n_estimators': [50, 100, 200],
                                                                    'algo__max_features': ['sqrt', 'auto'],
                                                                    'algo__min_samples_split': [2, 5, 10],
                                                                    'algo__min_samples_leaf': [2, 5],
                                                                    'algo__max_depth': [2, 4, 3, 5],
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
                self.cv_params['param_grid'] = self.algo_params['hyper_params_distribution']
            elif self.cv == 'RandomizedSearch':
                self.cv_params['param_distributions'] = self.algo_params['hyper_params_distribution']

            self.build_cv_pipeline()

    def build_preprocessor(self, use_base=True):
        # TODO: add extra feature if use_base=False
        if use_base:
            self.build_base_preprocessor(inplace=True)

    def build_algo(self):
        # TODO: add more choices for algorithm
        if self.algo_params['algo'] == 'RandomForestRegressor':
            self.algo = RandomForestRegressor(**self.algo_params['hyper_params'])

    def build_pipeline(self):
        self.pipeline = Pipeline([('preprocessor', self.preprocessor), ('algo', self.algo)])

    def build_cv_pipeline(self):
        if self.cv == 'GridSearch':
            self.cv_pipeline = GridSearchCV(**self.cv_params)
        elif self.cv == 'RandomizedSearch':
            self.cv_pipeline = RandomizedSearchCV(**self.cv_params)

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
        return self.pipeline.predict(X)








