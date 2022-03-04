import pandas as pd
import numpy as np
from sklearn import preprocessing

def preprocessing_min_max(df_test, df_train):
    """
    Give a scaler between 0 and 1 depending on the colum
    
    Parameters:
    df_test : pandas.DataFrame
        Data test frame we will processing
    df_train : pandas.DataFrame
        Data tain frame we will processing
    Return:
    df_test : pandas.DataFrame
        Data test processing
    df_train : pandas.DataFrame
        Data train processing
    """
    scaler = preprocessing.MinMaxScaler().fit(df_train);
    df_train[:] = scaler.transform(df_train[:])
    df_test[:] = scaler.transform(df_test[:])
    return df_test, df_train

def preprocessing_z_normal(df_test, df_train):
    """
    Give the z-normalisation depending on the colum
    
    Parameters:
    df_test : pandas.DataFrame
        Data test frame we will processing
    df_train : pandas.DataFrame
        Data tain frame we will processing
    Return:
    df_test : pandas.DataFrame
        Data test processing
    df_train : pandas.DataFrame
        Data train processing
    """
    scaler = preprocessing.StandardScaler().fit(df_train);
    df_train[:] = scaler.transform(df_train[:])
    df_test[:] = scaler.transform(df_test[:])
    return df_test, df_train

def preprocessing_polynomial(df, degree=2):
    """
    Give the polynomial features depending on the colum
    
    Parameters:
    df : pandas.DataFrame
        Data we will processing
    degree : int
        Degree of polynome we will
    Return:
    df : pandas.DataFrame
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
    d_train = {'col1': [1, 2, 3, 4, 5, 6, 7], 'col2': [2, 2, 2, 2, 2, 2, 2], 'col3': [8, 9, 10, 11, 12, 13, 14]}
    d_test = {'col1': [2, 5], 'col2': [2, 2], 'col3': [9, 12]}
    minmax_test, minmax_train = preprocessing_min_max(pd.DataFrame(data=d_test), pd.DataFrame(data=d_train))
    normal_test, normal_train = preprocessing_z_normal(pd.DataFrame(data=d_test), pd.DataFrame(data=d_train))
    polyno = preprocessing_polynomial(pd.DataFrame(data=d_test),3)
    print("Test preprocessing minmax...")
    print(minmax_test, "\n",minmax_train)
    print("Test preprocessing z-normal...")
    print(normal_test, "\n", normal_train)
    print("Test preprocessing polynomial...")
    np.set_printoptions(suppress=True)
    print(polyno)
    print("Test preprocessing original data...")
    print(pd.DataFrame(data=d_test), "\n", pd.DataFrame(data=d_train))
    