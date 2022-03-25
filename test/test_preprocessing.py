import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(1, './src')

import preprocessing_data as pr_da

def test_min_max_scaling_variant_1():
    """
    Test min_max_scaling of preprocessing
    Control:
        - Input value have not been changed
    """
    d_train = [[0, 0, 1, 2, 3, 4, 5, 6, 10], [0, 10, 5, 6, 2, 3, 1, 4, 0], [0, 10, 6, 8, 5, 6, 6, 10, 10]]
    d_test = [[2, 5, 5], [6, 6, 5], [8, 11, 10]]
    d_train = np.array(d_train).T
    d_test = np.array(d_test).T
    r_train = d_train.copy()
    r_test = d_test.copy()
    df_train = pd.DataFrame(data=d_train, columns=['col1','col2','col3'])
    df_test = pd.DataFrame(data=d_test, columns=['col1','col2','col3'])
    minmax_test, minmax_train = pr_da.min_max_scaling(df_test, df_train)
    
    assert np.all(r_train == d_train)
    assert np.all(r_test == d_test)

def test_min_max_scaling_variant_2():
    """
    Test min_max_scaling of preprocessing
    Control:
        - Output values are between 0 and 1
    """
    d_train = [[0, 0, 1, 2, 3, 4, 5, 6, 10], [0, 10, 5, 6, 2, 3, 1, 4, 0], [0, 10, 6, 8, 5, 6, 6, 10, 10]]
    d_test = [[2, 5, 5], [6, 6, 5], [8, 11, 10]]
    d_train = np.array(d_train).T
    d_test = np.array(d_test).T
    df_train = pd.DataFrame(data=d_train, columns=['col1','col2','col3'])
    df_test = pd.DataFrame(data=d_test, columns=['col1','col2','col3'])
    minmax_test, minmax_train = pr_da.min_max_scaling(df_test, df_train)
    
    assert (minmax_test.min().min() >= 0 and minmax_test.max().max() <= 1)
    assert (minmax_train.min().min() >= 0 and minmax_train.max().max() <= 1)

def test_min_max_scaling_variant_3():
    """
    Test min_max_scaling of preprocessing
    Control:
        - Output values are between 0 and 1 for negativ Input
    """
    d_train = [[-5, 0, -3, -2, -1, 0, 1, 5, 0], [0, -5, 1, -3, -2, -4, -1, 0, 5], [-5, -5, -2, -5, -3, -4, 0, 5, 5]]
    d_test = [[2, 5, -5], [-5, 1, -5], [-3, 6, -10]]
    d_train = np.array(d_train).T
    d_test = np.array(d_test).T
    df_train = pd.DataFrame(data=d_train, columns=['col1','col2','col3'])
    df_test = pd.DataFrame(data=d_test, columns=['col1','col2','col3'])
    minmax_test, minmax_train = pr_da.min_max_scaling(df_test, df_train)
    
    assert (minmax_test.min().min() >= 0 and minmax_test.max().max() <= 1)
    assert (minmax_train.min().min() >= 0 and minmax_train.max().max() <= 1)

def test_min_max_scaling_variant_4():
    """
    Test min_max_scaling of preprocessing
    Control:
        - Output values accordong to the input
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
    r_train = np.clip(r_train,0,1)
    r_test = np.clip(r_test,0,1)
    rf_train = pd.DataFrame(data=r_train)
    rf_test = pd.DataFrame(data=r_test)
    
    assert np.allclose(rf_train, minmax_train)
    assert np.allclose(rf_test, minmax_test)

def test_z_normalisation_variant_1():
    """
    Test z_normalisation of preprocessing
    Control:
        - Input value have not been changed
    """
    d_train = [[0, 0, 1, 2, 3, 4, 5, 6, 10], [0, 10, 5, 6, 2, 3, 1, 4, 0], [0, 10, 6, 8, 5, 6, 6, 10, 10]]
    d_test = [[2, 5, 5], [6, 6, 5], [8, 11, 10]]
    d_train = np.array(d_train).T
    d_test = np.array(d_test).T
    r_train = d_train.copy()
    r_test = d_test.copy()
    df_train = pd.DataFrame(data=d_train, columns=['col1','col2','col3'])
    df_test = pd.DataFrame(data=d_test, columns=['col1','col2','col3'])
    z_test, z_train = pr_da.z_normalisation(df_test, df_train)
    
    assert np.all(r_train == d_train)
    assert np.all(r_test == d_test)

def test_z_normalisation_variant_2():
    """
    Test z_normalisation of preprocessing
    Control:
        - Output means and variance values are 0 and 1
    """
    d_train = [[0, 0, 1, 2, 3, 4, 5, 6, 10], [0, 10, 5, 6, 2, 3, 1, 4, 0], [0, 10, 6, 8, 5, 6, 6, 10, 10]]
    d_test = [[2, 5, 5], [6, 6, 5], [8, 11, 10]]
    d_train = np.array(d_train).T
    d_test = np.array(d_test).T
    df_train = pd.DataFrame(data=d_train, columns=['col1','col2','col3'])
    df_test = pd.DataFrame(data=d_test, columns=['col1','col2','col3'])
    z_test, z_train = pr_da.z_normalisation(df_test, df_train)
    
    m = np.mean(z_train[:], axis=0)
    s = np.std(z_train[:], axis=0)

    assert np.allclose(m, 0.0)
    assert np.allclose(s, 1.0)

def test_z_normalisation_variant_3():
    """
    Test z_normalisation of preprocessing
    Control:
        - Output means and variance values are 0 and 1 for négative Input
    """
    d_train = [[-5, 0, -3, -2, -1, 0, 1, 5, 0], [0, -5, 1, -3, -2, -4, -1, 0, 5], [-5, -5, -2, -5, -3, -4, 0, 5, 5]]
    d_test = [[2, 5, -5], [-5, 1, -5], [-3, 6, -10]]
    d_train = np.array(d_train).T
    d_test = np.array(d_test).T
    df_train = pd.DataFrame(data=d_train, columns=['col1','col2','col3'])
    df_test = pd.DataFrame(data=d_test, columns=['col1','col2','col3'])
    z_test, z_train = pr_da.z_normalisation(df_test, df_train)
    
    m = np.mean(z_train[:], axis=0)
    s = np.std(z_train[:], axis=0)

    assert np.allclose(m, 0.0)
    assert np.allclose(s, 1.0)

def test_z_normalisation_variant_4():
    """
    Test z_normalisation of preprocessing
    Control:
        - Output values accordong to the input
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
    rf_train = pd.DataFrame(data=r_train, columns=['col1','col2','col3'])
    rf_test = pd.DataFrame(data=r_test, columns=['col1','col2','col3'])
    
    assert np.allclose(rf_train, z_train)
    assert np.allclose(rf_test, z_test)

def test_get_polynomial_variant_1():
    """
    Test get_polynomial_features of preprocessing
    Control:
        - Input value have not been changed
    """
    d = [[1, 2, 3], [2, 2, 2], [3, 4, 5]]
    d = np.array(d).T
    r = d.copy()
    df = pd.DataFrame(data=d, columns=['A', 'B', 'C'])
    p = pr_da.get_polynomial_features(df)

    assert np.all(d == r)

def test_get_polynomial_variant_2():
    """
    Test get_polynomial_features of preprocessing
    Control:
        - Output shape according to degree Input give
    """
    d = [[1, 2, 3], [2, 2, 2]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['A', 'B'])
    p1 = pr_da.get_polynomial_features(df,2)
    p2 = pr_da.get_polynomial_features(df,4)
    
    s1 = p1.shape
    s2 = p2.shape

    assert np.all(s1 == (3,6))
    assert np.all(s2 == (3,15))

def test_get_polynomial_variant_3():
    """
    Test get_polynomial_features of preprocessing
    Control:
        - Output shape according to nb of columns
    """
    d = [[1, 2, 3], [2, 2, 2]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['A', 'B'])
    p1 = pr_da.get_polynomial_features(df)
    d = [[1, 2, 3], [2, 2, 2], [3, 4, 5]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['A', 'B', 'C'])
    p2 = pr_da.get_polynomial_features(df)
    
    s1 = p1.shape
    s2 = p2.shape

    assert np.all(s1 == (3,6))
    assert np.all(s2 == (3,10))

def test_get_polynomial_variant_4():
    """
    Test get_polynomial_features of preprocessing
    Control:
        - Output shape according to test column
    """
    d = [[1, 2, 3], [2, 2, 2]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['A', 'B'])
    p1 = pr_da.get_polynomial_features(df, test=False)
    d = [[1, 2, 3], [2, 2, 2], [3, 4, 5]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['A', 'B', 'C'])
    p2 = pr_da.get_polynomial_features(df, test=True)
    
    s1 = p1.shape
    s2 = p2.shape
    
    assert np.all(s1 == (3,6))
    assert np.all(s2 == (3,7))

def test_get_polynomial_variant_5():
    """
    Test get_polynomial_features of preprocessing
    Control:
        - Output last column according to test column
    """
    d = [[1, 2, 3], [2, 2, 2]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['A', 'B'])
    p1 = pr_da.get_polynomial_features(df, test=False)
    d = [[1, 2, 3], [2, 2, 2], [3, 4, 5]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['A', 'B', 'C'])
    p2 = pr_da.get_polynomial_features(df, test=True)
    
    c1 = p1.columns
    c2 = p2.columns
    v1 = p1[c1[-1]]
    v2 = p2[c2[-1]]

    assert np.all(c1 == [0, 'A', 'B', 3, 4, 5])
    assert np.all(c2 == [0, 'A', 'B', 3, 4, 5, 'C'])
    assert np.all(v1 == [4, 4, 4])
    assert np.all(v2 == [3, 4, 5])

def test_get_polynomial_variant_6():
    """
    Test get_polynomial_features of preprocessing
    Control:
        - Output column name according to Input
    """
    d = [[1, 2, 3], [2, 2, 2]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['A', 'B'])
    p1 = pr_da.get_polynomial_features(df, test=False)
    p2 = pr_da.get_polynomial_features(df, test=True)
    p3 = pr_da.get_polynomial_features(df, degree=3, test=False)
    p4 = pr_da.get_polynomial_features(df, degree=3, test=True)
    
    c1 = p1.columns
    c2 = p2.columns
    c3 = p3.columns
    c4 = p4.columns

    assert np.all(c1 == [0, 'A', 'B', 3, 4, 5])
    assert np.all(c2 == [0, 'A', 2, 'B'])
    assert np.all(c3 == [0, 'A', 'B', 3, 4, 5, 6, 7, 8, 9])
    assert np.all(c4 == [0, 'A', 2, 3, 'B'])

def test_get_polynomial_variant_7():
    """
    Test get_polynomial_features of preprocessing
    Control:
        - Output values according to Input
    """
    d = [[1, 2, 3], [2, 2, 2]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['A', 'B'])
    p1 = pr_da.get_polynomial_features(df)
    d = [[1, 2, 3], [2, 2, 2], [3, 4, 5]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['A', 'B', 'C'])
    p2 = pr_da.get_polynomial_features(df)
    
    
    r1 = [[1,1,1], [1, 2, 3], [2, 2, 2], [1, 4, 9], [2, 4, 6], [4, 4, 4]]
    r1 = np.array(r1).T
    r2 = [[1,1,1], [1, 2, 3], [2, 2, 2], [3, 4, 5], [1, 4, 9], [2, 4, 6], [3, 8, 15], [4, 4, 4], [6, 8, 10], [9, 16, 25]]
    r2 = np.array(r2).T
    rf1 = pd.DataFrame(data=r1, columns=[0, 'A','B', 3, 4, 5])
    rf2 = pd.DataFrame(data=r2, columns=[0, 'A','B', 'C', 4, 5, 6, 7, 8, 9])
    

    assert np.allclose(rf1, p1)
    assert np.allclose(rf2, p2)
