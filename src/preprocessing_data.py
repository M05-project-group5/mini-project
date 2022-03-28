import pandas as pd
import numpy as np
from sklearn import preprocessing

def min_max_scaling(df_test, df_train):
    """
    Give a scaler between 0 and 1 depending on the colum
    
    Parameters
    ----------
    df_test : pandas.DataFrame
        Data test frame we will processing
    df_train : pandas.DataFrame
        Data tain frame we will processing
    Returns
    -------
    df_test : pandas.DataFrame
        Data test processing
    df_train : pandas.DataFrame
        Data train processing
    """
    scaler = preprocessing.MinMaxScaler(clip=True).fit(df_train)
    df_train[:] = scaler.transform(df_train[:])
    df_test[:] = scaler.transform(df_test[:])
    return df_test, df_train

def z_normalisation(df_test, df_train):
    """
    Give the z-normalisation depending on the colum
    
    Parameters
    ----------
    df_test : pandas.DataFrame
        Data test frame we will processing
    df_train : pandas.DataFrame
        Data tain frame we will processing
    Returns
    -------
    df_test : pandas.DataFrame
        Data test processing
    df_train : pandas.DataFrame
        Data train processing
    """
    scaler = preprocessing.StandardScaler().fit(df_train)
    df_train[:] = scaler.transform(df_train[:])
    df_test[:] = scaler.transform(df_test[:])
    return df_test, df_train

def get_polynomial_features(df, degree=2, test=False):
    """
    Give the polynomial features depending on the colum
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data we will processing
    degree : int
        Degree of polynome we will
    test : bool
        If the data who is give have result in the last column
    Returns
    -------
    df : pandas.DataFrame
        Data processing
    """
    head = df.columns.tolist()
    if test:
        last = df[head[-1:]]
        df = df[head[:-1]]
    df = preprocessing.PolynomialFeatures(degree=degree).fit_transform(df)
    df = pd.DataFrame(data=df)
    for i in range(0,len(head)-test):
        df.rename(columns={i+1: head[i]}, inplace=True)
    if test:
        df[head[-1:]] = last
    return df