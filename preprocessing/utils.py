from sklearn.pipeline import FeatureUnion


def generate_signal_feature_union(signal_config_list, signal_hyper_dict_prefix="signals"):
    # ToDo: add example!
    """Generate feature union estimator and hyper parameter dict from signal config list.
    :param list(dict) signal_config_list: List containing a dict referring to each transformer to add to
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

    for f in signal_config_list:
        if "desc" in f.keys():
            signal_desc = f["desc"]
        else:
            class_name = f["func"].__class__.__name__
            if class_name not in signal_count_dict.keys():
                signal_count_dict[class_name] = 0
            else:
                signal_count_dict[class_name] += 1
            signal_desc = f"{class_name}_{signal_count_dict[class_name]}"

        signal_pipe_list.append((signal_desc, f["func"]))

        for param_name, param_values in f["tuning_params"].items():
            signal_tuning_dict.update(
                {f"{signal_hyper_dict_prefix}__{signal_desc}__{param_name}": param_values}
            )

    signal_feature_union = FeatureUnion(signal_pipe_list)
    return signal_feature_union, signal_tuning_dict