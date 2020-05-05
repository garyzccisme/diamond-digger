from typing import Iterable, Dict, Optional

from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from preprocessing.imputer import DateImputer
from preprocessing.transformer import ColumnSelector, DateDeltaTransformer, DateSplitTransformer


def generate_tuning_dict(tune_params=None):
    """
    Helper function to generate hyper-parameters dict for cv pipeline. Simply reconstruct parameters name.
    Args:
        tune_params: Dict, {'prefix': {'param_name': distribution(List)}}

    Returns: Dict, {'prefix__param_name': distribution(List)}}

    """
    tuning_dict = {}
    if tune_params is not None:
        for prefix, params in tune_params.items():
            for name, distribution in params.items():
                tuning_dict['{}__{}'.format(prefix, name)] = distribution
    return tuning_dict


def generate_feature_union(transformer_dict_list):
    """
    Helper function to generate FeatureUnion given by a transformer_list.
    Args:
        transformer_dict_list: List, List containing a Dict referring to each transformer to add to the feature union
        estimator. Each Dict should contain three keys:
            {
                'prefix'(optional): String, The prefix name of the transformer(pipeline), if None then generate default.
                'transformer': Transformer or Pipeline, which's supposed to add into FeatureUnion.
                'tuning_params'(optional): Dict, {'param_name': distribution(list)}. Hyper-params prepared to tuned.
            }

    Returns: FeatureUnion, Tuning Parameters Dict

    """
    transformer_list = []
    transformer_count = {}
    tune_params = {}
    for transformer_dict in transformer_dict_list:
        if transformer_dict.get('transformer') is None:
            raise ValueError("Please make sure transformer is given.")
        if transformer_dict.get('prefix') is None:
            transformer_name = transformer_dict['transformer'].__class__.__name__
            transformer_count['transformer_name'] = transformer_count.get('transformer_name', 0) + 1
            transformer_dict['prefix'] = transformer_name + '_{}'.format(transformer_count['transformer_name'])
        transformer_list.append((transformer_dict['prefix'], transformer_dict['transformer']))
        if transformer_dict.get('tuning_params'):
            tune_params[transformer_dict['prefix']] = transformer_dict['tuning_params']
    tuning_dict = generate_tuning_dict(tune_params)
    feature_union = FeatureUnion(transformer_list=transformer_list)
    return feature_union, tuning_dict


def generate_cat_preprocessor(columns, imputer_strategy='most_frequent', encoder_type='Ordinal', tune_params=None):
    if encoder_type == 'Ordinal':
        encoder = OrdinalEncoder()
    elif encoder_type == 'OneHot':
        encoder = OneHotEncoder()
    else:
        raise ValueError("Invalid encoder_type, should be one of ['Ordinal', 'OneHot']")

    cat_preprocessor = Pipeline([
        ('selector', ColumnSelector(columns)),
        ('imputer', SimpleImputer(strategy=imputer_strategy)),
        ('encoder', encoder),
    ])
    feature_name = columns
    tuning_dict = generate_tuning_dict(tune_params)
    return cat_preprocessor, feature_name, tuning_dict


def generate_num_preprocessor(columns, imputer_strategy='median', scaler_type='Standard', tune_params=None):
    if scaler_type == 'Standard':
        scaler = StandardScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaler_type, should be one of ['Standard', 'MinMax']")

    num_preprocessor = Pipeline([
        ('selector', ColumnSelector(columns)),
        ('imputer', SimpleImputer(strategy=imputer_strategy)),
        ('scaler', scaler)
    ])
    feature_name = columns
    tuning_dict = generate_tuning_dict(tune_params)
    return num_preprocessor, feature_name, tuning_dict


