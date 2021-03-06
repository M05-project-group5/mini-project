#!/usr/bin/env python3
"""
This file contains the Machine Learning models to be tested in the pipeline.
"""
#Author:      Adrien Chassignet
#Co-authir:   Cédric Mariéthoz
#Date:        Mar 22 2022 
#Change date: Mars 3 2022 
#Version:     1.1

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

class ModifiedLinearReggression(LinearRegression):
    """ A modified inherited class that discards unwanted parameters. """

    def __init__(self, **args):
        """ Inherit all the methods and properties from the parent class 
        while using only expected parameters.
        """
        super().__init__()

class ModifiedDecisionTreeRegressor(DecisionTreeRegressor):
    """ A modified inherited class that discards unwanted parameters. """

    def __init__(self, **args):
        """ Inherit all the methods and properties from the parent class 
        while using only expected parameters.
        """
        try:
            rs = args['random_state']
        except KeyError:
            rs = None
        try:
            max_d = args['max_depth']
        except KeyError:
            max_d = None
        super().__init__(random_state=rs, max_depth=max_d)
