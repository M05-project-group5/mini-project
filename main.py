#!/usr/bin/env python3
"""
This script parse the command-line arguments to select block components of the 
model fitting pipeline.

@Author:    Adrien Chassignet, Cédric Mariéthoz
@Date:      Feb 28 2022 
@Version:   1.0
"""
import sys
sys.path.insert(1, 'src')

import os
import argparse
from download_datasets import download_wine, download_houses
from load_data import load_dataset
from split_data import split_data, split_x_y
from preprocessing_data import (get_polynomial_features,
                                min_max_scaling,
                                z_normalisation)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

redwine_file = "downloads/winequality-red.csv"
whitewine_file = "downloads/winequality-white.csv"
houses_file = "downloads/housing.data"

DATASETS = ['wine', 'houses']
PREPROCESSING = ['min-max', 'z-normalisation']
MODELS = ['linear-regression', 'regression-trees']
METRICS = ['mae']

def get_cl_args():
    """
    Parse the command line (sys.argv) to select elements of the pipeline.

    Return:
    args : class 'argparse.Namespace'
        The populated Namespace of the arguments. Each string argument is an 
        attribute of the namespace.
    """
    parser = argparse.ArgumentParser(description='Analyze datasets with ML '
                                    'regression techniques.')

    parser.add_argument('--dataset', action='store', choices=DATASETS,
                        help='Dataset to use between wine quality and Boston '
                        'house prices datasets.', default=DATASETS[0])
    parser.add_argument('--seed', action='store', type=int,
                        help='Seed for the pseudo-RNG used to split the data '
                        'and to initialize the models. '
                        'If no seed is given by the user, the system will be '
                        'fully random.', default=None)
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
    
def main(args):
    for arg in vars(args):
        print("{:11}->".format(arg), getattr(args, arg))

    # Load dataset (download it if not the already the case)
    if args.dataset == DATASETS[0]:
        if ((not os.path.isfile(redwine_file)) or
                (not os.path.isfile(whitewine_file))):
            download_wine()
            print("Wine dataset downloaded.")
    elif args.dataset == DATASETS[1]:
        if not os.path.isfile(houses_file):
            download_houses()
            print("Boston house prices dataset downloaded.")
    else:
        raise Exception("Unknown dataset for this pipeline.")
    
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
    else:
        raise Exception("Unknown pre-processing for this pipeline.")

    # Train model
    x_train, y_train = split_x_y(data_train)
    if args.model == MODELS[0]:
        model = LinearRegression()
    elif args.model == MODELS[1]:
        model = DecisionTreeRegressor(random_state=args.seed)
    else:
        raise Exception("Unknown model for this pipeline.")
    
    model.fit(x_train, y_train)

    # Analyze data
    x_test, y_test = split_x_y(data_test)
    # Inference
    y_train_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)
    #Evaluation
    if args.metrics == METRICS[0]:
        mae = mean_absolute_error(y_train, y_train_predict)
        print("On the train set: \nMean absolute error= ", mae)
        mae = mean_absolute_error(y_test, y_test_predict)
        print("On the test set: \nMean absolute error= ", mae)
    else:
        raise Exception("Unknown figure of merit for this pipeline.")


if __name__ == "__main__":
    # Parse command line arguments
    args = get_cl_args()
    main(args)