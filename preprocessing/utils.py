from typing import Iterable, Dict, Optional

from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from preprocessing.imputer import DateImputer
from preprocessing.transformer import ColumnSelector, DateDeltaTransformer, DateSplitTransformer


def generate_signal_feature_union(signal_list, signal_hyper_dict_prefix="signals"):
    # ToDo: add example!
    """Generate feature union estimator and hyper parameter dict from signal config list.
    :param list(dict) signal_list: List containing a dict referring to each transformer to add to
        the feature union estimator. Each dict should contain three keys -- 'desc' (optional), 'func',
        and 'tuning_params'. 'Desc' is a user-defined name of the transformer (optional), 'func' is a valid
        transformer function, and 'tuning_params' is a ``dict`` of parameters to pass to the transformer. If there
        are none, set to empty dict.
    :param str signal_hyper_dict_prefix: String to prepend to the signal hyper parameters, in order to use with
        `Pipeline` class. This string should match what the feature union is named in the pipeline. Defaults to
        `signals`.
    :returns: tuple: The initialized feature union and associated hyper-parameter dict.
    """
    signal_pipe_list = []
    signal_tuning_dict = {}
    signal_count_dict = {}

    for signal in signal_list:
        if "desc" in signal.keys():
            signal_desc = signal["desc"]
        else:
            class_name = signal["func"].__class__.__name__
            if class_name not in signal_count_dict.keys():
                signal_count_dict[class_name] = 0
            else:
                signal_count_dict[class_name] += 1
            signal_desc = '{}_{}'.format(class_name, signal_count_dict[class_name])

        signal_pipe_list.append((signal_desc, signal["func"]))

        for param_name, param_values in signal["tuning_params"].items():
            signal_tuning_dict.update(
                {'{}__{}__{}'.format(signal_hyper_dict_prefix, signal_desc, param_name): param_values}
            )
    signal_feature_union = FeatureUnion(signal_pipe_list)
    return signal_feature_union, signal_tuning_dict


def get_cat_preprocessor(columns, encoder_type, tune_params=None):

    if encoder_type == 'Ordinal':
        encoder = OrdinalEncoder()
    elif encoder_type == 'OneHot':
        encoder = OneHotEncoder()
    else:
        raise ValueError("Invalid encoder_type, should be one of ['Ordinal', 'OneHot']")

    cat_preprocessor = Pipeline([
        ('selector', ColumnSelector(columns)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder),
    ])

    tuning_dict = {}
    if tune_params is not None:
        for prefix, params in tune_params.items():
            for name, distribution in params.items():
                tuning_dict['{}__{}'.format(prefix, name)] = distribution

    return cat_preprocessor, tuning_dict


