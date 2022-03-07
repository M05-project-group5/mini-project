import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as decision_tree
from sklearn.linear_model import LinearRegression

def regression_trees(df_test, df_train, rs=None):
    """
    Initialize regession trees system with df_train and run it with df_test 
    
    Parameters:
    df_test : pandas.DataFrame
        Data test frame we will processing
    df_train : pandas.DataFrame
        Data tain frame we will processing
    rs : int
        Indication for the pseudo-random
        If it's None the function is random
    Return:
    df_result : pandas.DataFrame
        The data test result predict by regression trees
    """
    label = df_train.columns
    train_array = np.array(df_train[:])
    test_array = np.array(df_test[:])
    
    system = decision_tree(random_state=rs)
    system.fit(train_array[:,:-1], train_array[:,-1:])
    
    if(len(df_train.columns) != len(df_test.columns)):
        array_result = system.predict(test_array)
    else:
        array_result = system.predict(test_array[:,:-1])
    
    df_result =  pd.DataFrame(data=array_result, columns=label[-1:])
    return df_result

def linear_regression(df_test, df_train, rs=None):
    """
    Initialize linear regression system with df_train and run it with df_test 
    
    Parameters:
    df_test : pandas.DataFrame
        Data test frame we will processing
    df_train : pandas.DataFrame
        Data tain frame we will processing
    rs : int
        Indication for the pseudo-random
        If it's None the function is random
    Return:
    df_result : pandas.DataFrame
        The data test result predict by linear regression
    """
    label = df_train.columns
    train_array = np.array(df_train[:])
    test_array = np.array(df_test[:])
    
    system = LinearRegression(random_state=rs)
    system.fit(train_array[:,:-1], train_array[:,-1:])
    
    if(len(df_train.columns) != len(df_test.columns)):
        array_result = system.predict(test_array)
    else:
        array_result = system.predict(test_array[:,:-1])
    
    df_result =  pd.DataFrame(data=array_result, columns=label[-1:])
    return df_result

if __name__ == '__main__':
    print("Test system...")
    d_test = {'col1': [2, 5, 8], 'col2': [2, 2, 2], 'col3': [0, 0, 0]}
    d_test_curz = {'col1': [2, 8], 'col2': [2, 2]}
    d_train = {'col1': [1, 2, 3, 4, 6, 7], 'col2': [2, 2, 2, 2, 2, 2], 'col3': [8, 9, 10, 11, 13, 14]}
    df_test = pd.DataFrame(data=d_test)
    df_train = pd.DataFrame(data=d_train)
    df_test_curz = pd.DataFrame(data=d_test_curz)
    
    df_result_tree = regression_trees(df_test, df_train, 0)
    df_result_line = regression_trees(df_test, df_train, 0)
    print(df_result_tree)
    print(df_result_line)
    print(df_test)
    print(df_train)
    
    print("Test system with curz value...")
    df_result_tree = regression_trees(df_test_curz, df_train, 0)
    df_result_line = regression_trees(df_test_curz, df_train, 0)
    print(df_result_tree)
    print(df_result_line)
    print(df_test_curz)
    print(df_train)
    