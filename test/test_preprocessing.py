import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, './src')
sys.path.insert(0, '../src')

import preprocessing_data as pr_da

def test_min_max_scaling_positiv():
    """
    Test min_max_scaling of preprocessing for positiv values
    Control:
        - Output values according to the input
        - Output values are between 0 and 1
        - Input values are not modified by function
    """
    d_train = [[0, 0, 1, 2, 3, 4, 5, 6, 10], [0, 10, 5, 6, 2, 3, 1, 4, 0], [0, 10, 6, 8, 5, 6, 6, 10, 10]]
    d_test = [[2, 5, 5], [6, 6, 5], [8, 11, 10]]
    d_train = np.array(d_train).T
    d_test = np.array(d_test).T
    df_train = pd.DataFrame(data=d_train, columns=['col1','col2','col3'])
    df_test = pd.DataFrame(data=d_test, columns=['col1','col2','col3'])
    minmax_test, minmax_train = pr_da.min_max_scaling(df_test, df_train)
    
    r_train = d_train/10
    r_test = d_test/10
    np.clip(r_train,0,1)
    np.clip(r_test,0,1)
    rf_train = pd.DataFrame(data=r_train)
    rf_test = pd.DataFrame(data=r_test)
    
    assert np.allclose(rf_train, minmax_train)
    assert np.allclose(rf_test, minmax_test)
    assert (minmax_test.min().min() >= 0 and minmax_test.max().max() <= 1)
    assert (minmax_train.min().min() >= 0 and minmax_train.max().max() <= 1)
    assert np.all([[0, 0, 1, 2, 3, 4, 5, 6, 10], [0, 10, 5, 6, 2, 3, 1, 4, 0], [0, 10, 6, 8, 5, 6, 6, 10, 10]] == d_train.T)
    assert np.all([[2, 5, 5], [6, 1, 5], [8, 6, 10]] == d_test.T)

def test_min_max_scaling_negativ():
    """
    Test min_max_scaling of preprocessing for positiv and negativ values
    Control:
        - Output values according to the input
        - Output values are between 0 and 1
        - Input values are not modified by function
    """
    d_train = [[-5, 0, -4, -3, -2, -1, 0, 1, 5, 0], [0, -5, 0, 1, -3, -2, -4, -1, 0, 5], [-5, -5, -4, -2, -5, -3, -4, 0, 5, 5]]
    d_test = [[2, 5, 5], [-5, 1, 0], [-3, 6, 5]]
    d_train = np.array(d_train).T
    d_test = np.array(d_test).T
    df_train = pd.DataFrame(data=d_train, columns=['col1','col2','col3'])
    df_test = pd.DataFrame(data=d_test, columns=['col1','col2','col3'])
    minmax_test, minmax_train = pr_da.min_max_scaling(df_test, df_train)
    
    r_train = (d_train+5)/10
    r_test = (d_test+5)/10
    np.clip(r_train,0,1)
    np.clip(r_test,0,1)
    rf_train = pd.DataFrame(data=r_train)
    rf_test = pd.DataFrame(data=r_test)
    
    assert np.allclose(rf_train, minmax_train)
    assert np.allclose(rf_test, minmax_test)
    print(minmax_test)
    assert (minmax_test.min().min() >= 0.0 and minmax_test.max().max() <= 1.0)
    assert (minmax_train.min().min() >= 0.0 and minmax_train.max().max() <= 1.0)
    assert np.all([[-5, 0, -4, -3, -2, -1, 0, 1, 5, 0], [0, -5, 0, 1, -3, -2, -4, -1, 0, 5], [-5, -5, -4, -2, -5, -3, -4, 0, 5, 5]] == d_train.T)
    assert np.all([[2, 5, 5], [-5, 1, 0], [-3, 6, 5]] == d_test.T)

def test_z_normalisation_positiv():
    """
    Test z-normalisation of preprocessing for positiv values
        - Output values according to the input
        - Output values are between -1 and 1
        - Input values are not modified by function
    """
    d_train = [[0, 0, 1, 2, 3, 4, 5, 6, 10], [0, 10, 5, 6, 2, 3, 1, 4, 0], [0, 10, 6, 8, 5, 6, 6, 10, 10]]
    d_test = [[2, 5, 5], [6, 6, 5], [8, 11, 10]]
    d_train = np.array(d_train).T
    d_test = np.array(d_test).T
    df_train = pd.DataFrame(data=d_train, columns=['col1','col2','col3'])
    df_test = pd.DataFrame(data=d_test, columns=['col1','col2','col3'])
    z_test, z_train = pr_da.z_normalisation(df_test, df_train)
    
    m = np.mean(d_train, axis=0)
    s = np.std(d_train, axis=0)
    r_train = (d_train-m)/s
    r_test = (d_test-m)/s
    r_train = np.clip(r_train,-1,1)
    r_test = np.clip(r_test,-1,1)
    rf_train = pd.DataFrame(data=r_train, columns=['col1','col2','col3'])
    rf_test = pd.DataFrame(data=r_test, columns=['col1','col2','col3'])
    
    assert np.allclose(rf_train, minmax_train)
    assert np.allclose(rf_test, minmax_test)
    assert (minmax_test.min().min() >= -1.0 and minmax_test.max().max() <= 1.0)
    assert (minmax_train.min().min() >= -1.0 and minmax_train.max().max() <= 1.0
    assert np.all([[0, 0, 1, 2, 3, 4, 5, 6, 10], [0, 10, 5, 6, 2, 3, 1, 4, 0], [0, 10, 6, 8, 5, 6, 6, 10, 10]] == d_train.T)
    assert np.all([[2, 5, 5], [6, 6, 5], [8, 11, 10]] == d_test.T)
    
def test_z_normalisation_negativ():
    """
    Test z-normalisation of preprocessing for positv and negativ values
        - Output values according to the input
        - Output values are between -1 and 1
        - Input values are not modified by function
    """
    d_train = [[-5, 0, -4, -3, -2, -1, 0, 1, 5, 0], [0, -5, 0, 1, -3, -2, -4, -1, 0, 5], [-5, -5, -4, -2, -5, -3, -4, 0, 5, 5]]
    d_test = [[2, 5, 5], [-5, 1, 0], [-3, 6, 5]]
    d_train = np.array(d_train).T
    d_test = np.array(d_test).T
    df_train = pd.DataFrame(data=d_train, columns=['col1','col2','col3'])
    df_test = pd.DataFrame(data=d_test, columns=['col1','col2','col3'])
    z_test, z_train = pr_da.z_normalisation(df_test, df_train)
    
    m = np.mean(d_train, axis=0)
    s = np.std(d_train, axis=0)
    r_train = (d_train-m)/s
    r_test = (d_test-m)/s
    r_train = np.clip(r_train,-1,1)
    r_test = np.clip(r_test,-1,1)
    rf_train = pd.DataFrame(data=r_train, columns=['col1','col2','col3'])
    rf_test = pd.DataFrame(data=r_test, columns=['col1','col2','col3'])
    
    assert np.allclose(rf_train, minmax_train)
    assert np.allclose(rf_test, minmax_test)
    assert (minmax_test.min().min() >= -1.0 and minmax_test.max().max() <= 1.0)
    assert (minmax_train.min().min() >= -1.0 and minmax_train.max().max() <= 1.0
    assert np.all([[0, 0, 1, 2, 3, 4, 5, 6, 10], [0, 10, 5, 6, 2, 3, 1, 4, 0], [0, 10, 6, 8, 5, 6, 6, 10, 10]] == d_train.T)
    assert np.all([[2, 5, 5], [6, 6, 5], [8, 11, 10]] == d_test.T)

def test_get_polynomial_features():
    """
    Test get_polynomial_features of preprocessing
        - Output values according to the input
        - Columns names rigth about test or not
        - Degree of the function according to input
    """
    d1 = [[1, 2, 3], [2, 2, 2]]
    d2 = [[1, 2, 3], [2, 2, 2], [3, 4, 5]]
    d3 = [[1, 2, 3], [2, 2, 2]]
    d1 = np.array(d1).T
    d2 = np.array(d2).T
    d3 = np.array(d3).T
    df1 = pd.DataFrame(data=d1, columns=['A','B'])
    df2 = pd.DataFrame(data=d2, columns=['A','B','Y'])
    df3 = pd.DataFrame(data=d3, columns=['A','B'])
    p1 = pr_da.get_polynomial_features(df1)
    p2 = pr_da.get_polynomial_features(df2,test=True)
    p3 = pr_da.get_polynomial_features(df3,degree=3)
    
    r1 = [[1,1,1], [1, 2, 3], [2, 2, 2], [1, 4, 9], [2, 4, 6], [4, 4, 4]]
    r1 = np.array(r1).T
    r2 = [[1,1,1], [1, 2, 3], [2, 2, 2], [1, 4, 9], [2, 4, 6], [4, 4, 4], [3, 4, 5]]
    r2 = np.array(r2).T
    r3 = [[1,1,1], [1, 2, 3], [2, 2, 2], [1, 4, 9], [2, 4, 6], [4, 4, 4], [1, 8, 27], [2, 8, 18], [4, 8, 12], [8, 8, 8]]
    r3 = np.array(r3).T
    rf1 = pd.DataFrame(data=r1, columns=[0, 'A','B', 3, 4, 5])
    rf2 = pd.DataFrame(data=r2, columns=[0, 'A','B', 3, 4, 5, 'Y'])
    rf3 = pd.DataFrame(data=r3, columns=[0, 'A','B', 3, 4, 5, 6, 7, 8, 9])
    
    assert np.allclose(rf1, p1)
    assert np.allclose(rf2, p2)
    assert np.allclose(rf3, p3)
    assert (rf1.columns == p1.columns).all()
    assert (rf2.columns == p2.columns).all()
    assert (rf3.columns == p3.columns).all()
