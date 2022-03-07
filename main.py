#!/usr/bin/env python3
"""
This script parse the command-line arguments to select block components of the 
model fitting pipeline.

@Author:    Adrien Chassignet
@Date:      Feb 28 2022 
@Version:   1.0
"""
from statistics import mean
import sys
sys.path.insert(1, 'src')

import argparse
from load_data import load_dataset
from split_data import split_data
from preprocessing_data import (get_polynomial_features,
                                min_max_scaling,
                                z_normalisation)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DATASETS = ['wine', 'houses']
SEEDS = range(4)
PREPROCESSING = ['min-max', 'z-normalisation']
MODELS = ['linear-regression', 'regression-trees']
METRICS = ['mae']

def get_cl_args():
    """
    Parse the command line (sys.argv) to select elements of the pipeline.

    Return:
    args:   class 'argparse.Namespace'
        The populated Namespace of the arguments. Each string argument is an 
        attribute of the namespace.
    """
    parser = argparse.ArgumentParser(description='Analyze datasets with ML '
                                    'regression techniques.')

    parser.add_argument('--dataset', action='store', choices=DATASETS,
                        help='Dataset to use between wine quality and Boston '
                        'house prices datasets.', default=DATASETS[0])
    parser.add_argument('--seed', action='store', choices=SEEDS, type=int,
                        help='Seed for the pseudo-RNG used to split the data',
                        default=SEEDS[0])
    parser.add_argument('--scaling', action='store', choices=PREPROCESSING,
                        help='Select the scaling pre-processing technique to '
                        'apply to the features.', default=PREPROCESSING[0])
    parser.add_argument('--polynomial', action='store_true', 
                        help='Use polynomial features instead of orginial ones '
                        'for pre-processing')
    parser.add_argument('--model', action='store', choices=MODELS,
                        help='Select the ML model that will be used to analyze '
                        'the data.', default=MODELS[0])
    parser.add_argument('--metrics', action='store', choices=METRICS,
                        help='Choose the metrics used as a measure of success '
                        'of the chosen model.', default=METRICS[0])

    return parser.parse_args()
    

if __name__ == "__main__":
    # Parse command line arguments
    args = get_cl_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Load dataset
    # TODO: check if files has been downloaded and download if not the case
    data = load_dataset(name=args.dataset)

    # Split data
    data_train, data_test = split_data(data, rs=args.seed)

    # Pre-processing
    if args.polynomial:
        data_train = get_polynomial_features(data_train)
        data_test = get_polynomial_features(data_test)
    if args.scaling == PREPROCESSING[0]:
        data_test, data_train = min_max_scaling(data_test, data_train)
    elif args.scaling == PREPROCESSING[1]:
        data_test, data_train = z_normalisation(data_test, data_train)

    # Train model
    x_train = data_train.iloc[:, 0:-1]
    y_train = data_train.iloc[:, -1]
    
    if args.model == MODELS[0]:
        model = LinearRegression()
    elif args.model == MODELS[1]:
        print('Regression trees not implemented yet.')
        sys.exit()
        # model = RegressionTrees()
    
    model.fit(x_train, y_train)

    # Analyze data
    x_test = data_test.iloc[:, 0:-1]
    y_test = data_test.iloc[:, -1]
    # Inference
    y_test_predict = model.predict(x_test)
    #Evaluation
    if args.metrics == METRICS[0]:
        mae = mean_absolute_error(y_test, y_test_predict)
        print("Mean absolute error= ", mae)