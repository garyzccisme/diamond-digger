from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from preprocessing.imputer import DateImputer
from model.base import BaseModel


class DiamondPricer(BaseModel):
    def __init__(self, preprocessor_params=None, algo_params=None, cv_params=None):
        super().__init__(preprocessor_params, algo_params, cv_params)

    def build_preprocessor(self, use_base=True):

        # Initialize preprocessor_params with default
        if self.preprocessor_params is None:
            self.preprocessor_params = {
                'target_col': ['Price'],
                'cat_col': ['Shape', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Fluorescence', 'Culet'],
                'cat_encoder': OrdinalEncoder(),
                'cat_imputer': SimpleImputer(strategy='most_frequent'),
                'num_col': ['Carat', 'Depth', 'Table', 'L/W'],
                'num_imputer': SimpleImputer(strategy="median"),
                'num_scaler': StandardScaler(),
                'date_col': ['First Available Date', 'Last Available Date', 'Delivery Date'],
                'date_imputer': DateImputer(),
                'date_splitter': ['First Available Date'],
                'date_delta': ['deliver_days', 'in_stock_days'],
            }

        if use_base:
            self.build_base_preprocessor(inplace=True)
        # TODO: add extra feature if use_base=False









