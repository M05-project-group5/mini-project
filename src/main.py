#!/usr/bin/env python3
"""
This script parse the command-line arguments to select block components of the 
model fitting pipeline.
"""
#Author:      Adrien Chassignet
#Co-authir:   Cédric Mariéthoz
#Date:        Feb 28 2022
#Change date: Mar 3 2022 
#Version:     1.1

import sys
import os
import argparse
from download_datasets import download_wine, download_houses
from src.load_data import load_dataset
from src.split_data import split_data, split_x_y
from src.preprocessing_data import (get_polynomial_features,
                                min_max_scaling,
                                z_normalisation)
from src.models import ModifiedLinearReggression, ModifiedDecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

DATASETS = {'wine': download_wine, 'houses': download_houses}
PREPROCESSING = {'min-max': min_max_scaling, 'z-normalisation': z_normalisation}
MODELS =   {'linear-regression': ModifiedLinearReggression,
            'regression-trees': ModifiedDecisionTreeRegressor}
METRICS = {'mae': mean_absolute_error}
#""" Documentation of global variables"""

def get_cl_args(args=sys.argv[1:]):
    """
    Parse the command line (sys.argv) to select elements of the pipeline.

    Returns
    -------
    args : class 'argparse.Namespace'
        The populated Namespace of the arguments. Each string argument is an 
        attribute of the namespace.
    """
    parser = argparse.ArgumentParser(description='Analyze datasets with ML '
                                    'regression techniques.')

    parser.add_argument('-d', '--dataset', action='store', choices=DATASETS.keys(),
                        help='Dataset to use between wine quality and Boston '
                        'house prices datasets.', default=list(DATASETS)[0])
    parser.add_argument('--seed', action='store', type=int,
                        help='Seed for the pseudo-RNG used to split the data '
                        'and to initialize the models. '
                        'If no seed is given by the user, the system will be '
                        'fully random.', default=None)
    parser.add_argument('--scaling', action='store', choices=PREPROCESSING.keys(),
                        help='Select the scaling pre-processing technique to '
                        'apply to the features.', default=list(PREPROCESSING)[0])
    parser.add_argument('--polynomial', action='store', type=int, nargs='?',
                        help='Use polynomial features with given degree instead'
                        ' of orginial ones for pre-processing', const=2,
                        default=None)
    parser.add_argument('-m', '--model', action='store', choices=MODELS.keys(),
                        help='Select the ML model that will be used to analyze '
                        'the data.', default=list(MODELS)[0])
    parser.add_argument('--depth', action='store', type=int, default=None,
                        help='Maximum depth of the tree when using regression '
                        'trees.')
    parser.add_argument('--metrics', action='store', choices=METRICS.keys(),
                        help='Choose the metrics used as a measure of success '
                        'of the chosen model.', default=list(METRICS)[0])

    return parser.parse_args(args)
    
def main(args):
    for arg in vars(args):
        print("{:11}->".format(arg), getattr(args, arg))

    # Load dataset (download it if not the already the case)
    try:
        DATASETS[args.dataset]()
    except KeyError:
        raise RuntimeError(f"{args.dataset} is invalid for --dataset. Choose "
                            f"between ({', '.join(DATASETS.keys())})")
    
    data = load_dataset(name=args.dataset)
    data_train, data_test = split_data(data, rs=args.seed)

    # Pre-processing
    if args.polynomial:
        data_train = get_polynomial_features(data_train, degree=args.polynomial)
        data_test = get_polynomial_features(data_test, degree=args.polynomial)

    try:
        data_test, data_train = PREPROCESSING[args.scaling](data_test, data_train)
    except KeyError:
        raise RuntimeError(f"{args.scaling} is invalid for --scaling. Choose "
                            f"between ({', '.join(PREPROCESSING.keys())})")

    # Train model
    x_train, y_train = split_x_y(data_train)
    try:
        model = MODELS[args.model](random_state=args.seed, max_depth=args.depth)
        model.fit(x_train, y_train)
    except KeyError:
        raise RuntimeError(f"{args.model} is invalid for --model. Choose "
                            f"between ({', '.join(MODELS.keys())})")
    
    # Analyze data
    x_test, y_test = split_x_y(data_test)
    # Inference
    y_train_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)
    #Evaluation
    try:
        print('===============================')
        train_figure_of_merit = METRICS[args.metrics](y_train, y_train_predict)
        print(f"On the train set: \n{args.metrics} = {train_figure_of_merit}")
        test_figure_of_merit = METRICS[args.metrics](y_test, y_test_predict)
        print(f"On the test set: \n{args.metrics} = {test_figure_of_merit}")
    except KeyError:
        raise RuntimeError(f"{args.metrics} is invalid for --metrics. Choose "
                            f"between ({', '.join(METRICS.keys())})")


if __name__ == "__main__":  # pragma: no cover
    # Parse command line arguments
    args = get_cl_args()
    main(args)