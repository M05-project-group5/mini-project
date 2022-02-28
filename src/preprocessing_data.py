import pandas as pd
import numpy as np
from sklearn import preprocessing

def preprocessing_min_max(df):
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

def preprocessing_z_normal(df):
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

def preprocessing_polynomial(df, degree=2):
    """
    Give the polynomial features depending on the colum
    
    Parameters:
    df : pandas.DataFrame
        Data we will processing
    degree : int
        Degree of polynome we will
    Return:
        pandas.DataFrame
        Data processing
    """
    head = df.columns.tolist()
    df = preprocessing.PolynomialFeatures(degree=degree).fit_transform(df[:])
    df = pd.DataFrame(data=df)
    for i in range(0,len(head)):
        df.rename(columns={i+1: head[i]}, inplace=True)
    return df

if __name__ == '__main__':
    print("Test preprocessing data...")
    d = {'col1': [1, 2, 3, 4, 5, 6, 7], 'col2': [2, 2, 2, 2, 2, 2, 2], 'col3': [8, 9, 10, 11, 12, 13, 14]}
    minMax = preprocessing_min_max(pd.DataFrame(data=d))
    normal = preprocessing_z_normal(pd.DataFrame(data=d))
    polyno = preprocessing_polynomial(pd.DataFrame(data=d),3)
    print("Test preprocessing minmax...")
    print(minMax)
    print("Test preprocessing z-normal...")
    print(normal)
    print("Test preprocessing polynomial...")
    np.set_printoptions(suppress=True)
    print(polyno)
    print("Test preprocessing original data...")
    print(pd.DataFrame(data=d))
    