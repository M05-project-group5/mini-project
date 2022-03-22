"""
This file contains the Machine Learning models to be tested in the pipeline.

@Author:    Adrien Chassignet
@Date:      Mar 22 2022 
@Version:   1.0
"""
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
        rs = args['random_state']
        super().__init__(random_state=rs)