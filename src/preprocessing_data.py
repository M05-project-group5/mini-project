import pandas as pd
import numpy as np
from sklearn import preprocessing


def preprocessing_minMax(df):
    """
    Give a scaler between 0 and 1 depending on the colum
    
    Parameters:
    df : pandas.DataFrame
        Data we will processing
    Return:
        pandas.DataFrame
        Data processing
    """
    df[:] = preprocessing.MinMaxScaler().fit_transform(df[:])
    return df


def preprocessing_zNormal(df):
    """
    Give the z-normalisation depending on the colum
    
    Parameters:
    df : pandas.DataFrame
        Data we will processing
    Return:
        pandas.DataFrame
        Data processing
    """
    df[:] = preprocessing.StandardScaler().fit_transform(df[:])
    return df


if __name__ == '__main__':
    print("Test preprocessing data...")
    d = {'col1': [1, 2, 3, 4, 5, 6, 7], 'col2': [2, 2, 2, 2, 2, 2, 2], 'col3': [8, 9, 10, 11, 12, 13, 14]}
    minMax = preprocessing_minMax(pd.DataFrame(data=d))
    normal = preprocessing_zNormal(pd.DataFrame(data=d))
    print("Test preprocessing minmax...")
    print(minMax)
    print("Test preprocessing z-normal...")
    print(normal)
    print("Test preprocessing original data...")
    print(pd.DataFrame(data=d))
    