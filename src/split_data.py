import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def splitData(df, testSize=0.5, rs=None):
    """
    Split in two a data frame
    The random_state is used for change the shuffling before the split
    
    Parameters:
    df : pandas.DataFrame
        Data we will splitting
    testSize : float or int
        Value between 0 and 1 for size of test split
    rs : int
        Indication for the pseudo-random split
        If it's None the function is random
    Return:
    train : pandas.DataFrame
        Train split data
    test : pandas.DataFrame
        Test split data
    """
    train, test = train_test_split(df, test_size=testSize, random_state=rs)
    return train, test


if __name__ == '__main__':
    print("Test split data...")
    d = {'col1': [1, 2, 3, 4, 5, 6, 7], 'col2': [2, 2, 2, 2, 2, 2, 2]}
    train, test = splitData(pd.DataFrame(data=d), rs=50)
    print(train)
    print(test)
    print("Test pseudo-random split...")
    train, test = splitData(pd.DataFrame(data=d), rs=50)
    print(train)
    print(test)
    print("Test random split...")
    train, test = splitData(pd.DataFrame(data=d))
    print(train)
    print(test)
    print("Original data...")
    print(pd.DataFrame(data=d))