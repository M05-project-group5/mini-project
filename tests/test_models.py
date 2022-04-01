import pytest
import sys
sys.path.insert(1, './src')

from models import ModifiedLinearReggression, ModifiedDecisionTreeRegressor

def test_init_mod_linear_regression_unwanted_params():
    """ Test if LinearRegression is correctly initiated even with unwanted 
    parameters.
    """
    ModifiedLinearReggression(random_state=0, dummy='testing')

def test_init_mod_regression_trees_missing_random_state():
    """ Test if DecisionTreeRegressor is correctly initiated even with missing 
    random_state parameter.
    """
    model = ModifiedDecisionTreeRegressor()
    assert model.random_state == None

def test_init_mod_regression_trees_missing_max_depth():
    """ Test if DecisionTreeRegressor is correctly initiated even with missing 
    max_depth parameter.
    """
    model = ModifiedDecisionTreeRegressor()
    assert model.max_depth == None

def test_init_mod_regression_trees_given_random_state():
    """ Test if DecisionTreeRegressor is correctly initiated with given 
    random_state parameter.
    """
    model = ModifiedDecisionTreeRegressor(random_state=0)
    assert model.random_state == 0

def test_init_mod_regression_trees_given_max_depth():
    """ Test if DecisionTreeRegressor is correctly initiated with given 
    max_depth parameter.
    """
    model = ModifiedDecisionTreeRegressor(max_depth=10)
    assert model.max_depth == 10

def test_init_mod_regression_trees_unwanted_params():
    """ Test if DecisionTreeRegressor is correctly initiated even with unwanted 
    parameters.
    """
    ModifiedDecisionTreeRegressor(dummy='testing')
