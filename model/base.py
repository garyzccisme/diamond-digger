from sklearn.base import BaseEstimator


class BaseModel(BaseEstimator):
    def __init__(self, preprocessor_params=None, algo_params=None, cv_params=None):
        self.preprocessor_params = preprocessor_params
        self.algo_params = algo_params
        self.cv_params = cv_params

        self.preprocessor = None
        self.algo = None
        self.pipeline = None

    def build_preprocessor(self):
        pass

    def load_algo(self):
        pass

    def build_pipeline(self):
        pass
