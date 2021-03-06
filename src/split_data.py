#!/usr/bin/env python3
"""
This script has the function of splitting the data into the test and training stack.

We have a function to split the data into two sets.
The second divides the data between the parameters and the result.
"""
#Author:      Cédric Mariéthoz
#Co-authir:   Adrien Chassignet
#Date:        Feb 28 2022
#Change date: Mar 3 2022 
#Version:     1.1

import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.5, rs=None):
    """Split in two a data frame.
    
    The random_state is used for change the shuffling before the split.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data we will splitting
    testSize : float or int
        Value between 0 and 1 for size of test split
    rs : int
        Indication for the pseudo-random split
        If it's None the function is random
    
    Returns
    -------
    train : pandas.DataFrame
        Train split data
    test : pandas.DataFrame
        Test split data
    """
    train, test = train_test_split(df, test_size=test_size, random_state=rs)
    return train, test

def split_x_y(data):
    """Split the data between x, the parameters, and y, the result.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be separated between x parameters columns and y result column
    
    Returns
    -------
    x : pandas.DataFrame
        The parameters column of the DataFrame data (all but the last column)
    y : pandas.DataFrame
        The result column of the DataFrame data (the last column)
    """
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y
