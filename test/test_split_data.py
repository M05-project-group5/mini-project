import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(1, './src')

import split_data as sp_da

def test_split_data_variant_1():
    """
    Test of split for the values who was split
    Control:
        - Check if all data are in the output
    """
    d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['col1','col2','col3'])
    df_test1, df_train1 = sp_da.split_data(df)
    df1 = pd.concat([df_test1, df_train1], axis=0)
    dfr = pd.merge(df, df1, how='left', indicator='same')
    r = dfr['same'] == 'both'
    
    assert np.all(r)
    
    
def test_split_data_variant_2():
    """
    Test of split for the rigth size
    Control:
        - The sum of size of the two output is the size of input
        - Output df_test size
        - Output df_train size
    """
    d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['col1','col2','col3'])
    df_train1, df_test1 = sp_da.split_data(df)
    df_train2, df_test2 = sp_da.split_data(df,test_size=0.3)
    df_train3, df_test3 = sp_da.split_data(df,test_size=0.9)

    assert df_train1.shape[0] + df_test1.shape[0] == df.shape[0]
    assert df_train2.shape[0] + df_test2.shape[0] == df.shape[0]
    assert df_train3.shape[0] + df_test3.shape[0] == df.shape[0]
    assert df_test1.shape[0] == 5
    assert df_test2.shape[0] == 3
    assert df_test3.shape[0] == 9
    assert df_train1.shape[0] == 5
    assert df_train2.shape[0] == 7
    assert df_train3.shape[0] == 1
    
def test_split_data_variant_3():
    """
    Test of split for random and pseudo-random
    Control:
        - If two random are not the equals
        - If two with same pseudo-random are same
        - If two with different pseudo-random are not equals
    """
    d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['col1','col2','col3'])
    df_train1, df_test1 = sp_da.split_data(df)
    df_train2, df_test2 = sp_da.split_data(df, rs=None)
    df_train3, df_test3 = sp_da.split_data(df, rs=5)
    df_train4, df_test4 = sp_da.split_data(df, rs=5)
    df_train5, df_test5 = sp_da.split_data(df, rs=10)
    
    assert not df_test1.equals(df_test2)
    assert not df_train1.equals(df_train2)
    assert df_test3.equals(df_test4)
    assert df_train3.equals(df_train4)
    assert not df_test4.equals(df_test5)
    assert not df_train4.equals(df_train5)

def test_split_x_y():
    """
    Test of split_x_y
    Control:
        - The split_x_y don't change values or order
    """
    d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    d = np.array(d).T
    df = pd.DataFrame(data=d, columns=['col1','col2','col3'])
    x, y = sp_da.split_x_y(df)
    xy = pd.concat([x, y], axis=1)

    assert xy.equals(df)
    