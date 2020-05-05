from typing import Iterable, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from preprocessing.imputer import DateImputer
from preprocessing.transformer import ColumnSelector, DateDeltaTransformer, DateSplitTransformer


def generate_tuning_dict(tune_params: Dict = None):
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


def generate_feature_union(transformer_dict_list: List):
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

    Returns: FeatureUnion, Tuning Parameters Dict {'prefix__param_name': distribution(list)}.

    """
    transformer_list = []
    tune_params = {}
    transformer_count = {}

    for transformer_dict in transformer_dict_list:
        if transformer_dict.get('transformer') is None:
            raise ValueError("Please make sure transformer is given.")
        # If prefix is None, then generate a default prefix by class name and order.
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
    """
    Helper function to generate categorical features preprocessor pipeline [ColumnSelector, SimpleImputer, Encoder].

    Args:
        columns: Iterable, List of categorical columns supposed to be fed into model.
        imputer_strategy: String, `strategy` parameter of SimpleImputer. Default is 'most_frequent'.
        encoder_type: String, if 'Ordinal' then use OrdinalEncoder, if 'OneHot' then use OneHotEncoder.
        tune_params: Dict, tuning parameters dict, the keys should be in ['selector', 'imputer', 'encoder'],
            which are steps of the Pipeline.

    Returns: preprocessor, feature names, tuning hyper-parameters. Pipeline, List, Dict.

    """
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
    feature_name = list(columns)
    tuning_dict = generate_tuning_dict(tune_params)
    return cat_preprocessor, feature_name, tuning_dict


def generate_num_preprocessor(columns, imputer_strategy='median', scaler_type='Standard', tune_params=None):
    """
    Helper function to generate numerical features preprocessor pipeline [ColumnSelector, SimpleImputer, Scaler].

    Args:
        columns: Iterable, List of numerical columns supposed to be fed into model.
        imputer_strategy: String, `strategy` parameter of SimpleImputer. Default is 'median'.
        scaler_type: String, if 'Standard' then use StandardScaler, if 'MinMax' then use MinMaxScaler.
        tune_params: Dict, tuning parameters dict, the keys should be in ['selector', 'imputer', 'scaler'],
            which are steps of the Pipeline.

    Returns: preprocessor, feature names, tuning hyper-parameters. Pipeline, List, Dict.

    """
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
    feature_name = list(columns)
    tuning_dict = generate_tuning_dict(tune_params)
    return num_preprocessor, feature_name, tuning_dict


def generate_date_preprocessor(split_cols, delta_types, imputer_strategy=None):
    """
    Helper function to generate numerical features preprocessor pipeline [ColumnSelector, DateImputer, DateTransformer].
    tune_params is invalid input here since there aren't tunable parameters in Pipeline currently.

    Args:
        split_cols: Iterable, columns put into DateSplitTransformer to split.
        delta_types: Iterable, each element should be valid parameter of `DateDeltaTransformer.delta_type`.
        imputer_strategy: Dict, parameters of DateImputer.

    Returns: preprocessor, feature names. Pipeline, List

    """
    if imputer_strategy is None:
        imputer_strategy = {
            'first_available_date': 'earliest',
            'last_available_date': 'latest',
            'deliver_date': 'latest'
        }

    splitter = [{
        'prefix': col + 'split', 'transformer': DateSplitTransformer(date_type=col),
    } for col in split_cols]
    delta = [{
        'prefix': delta_type, 'transformer': DateDeltaTransformer(delta_type=delta_type),
    } for delta_type in delta_types]

    date_feature_union, _ = generate_feature_union(splitter + delta)
    date_preprocessor = Pipeline([
            ('selector', ColumnSelector(['First Available Date', 'Last Available Date', 'Delivery Date'])),
            ('imputer', DateImputer(**imputer_strategy)),
            ('date_feature', date_feature_union)
        ])
    feature_name = np.array([trans['transformer'].split_feature_name for trans in splitter]).flatten().tolist()
    feature_name += delta_types
    return date_preprocessor, feature_name
