import sys
import pandas as pd

VERBOSE = True

redwine_file = "downloads/winequality-red.csv"
whitewine_file = "downloads/winequality-white.csv"
houses_file = "downloads/housing.data"

def read_data(func, *args, **kwargs):
    """
    Read data file with given parsing function and arguments.
    
    Return:
    df : pandas.DataFrame
        Loaded data from the given file in arguments.

    Example:
        df = read_data(pd.read_csv, filepath_or_buffer=filename.csv, sep=';')
    """
    return func(*args, **kwargs)

def load_wine_dataset():
    """
    Load the wine quality dataset from "downloads/winequality-red.csv" and 
    "downloads/winequality-white.csv". Both dataset are merged.

    Return:
    df : pandas.DataFrame
        Wine quality loaded data
    """
    # Read the two type of wine datasets
    redwine_df = read_data(pd.read_csv, redwine_file, sep=';')
    whitewine_df = read_data(pd.read_csv, whitewine_file, sep=';')
    # Concatenate the two type of wine datasets
    wine_df = pd.concat([redwine_df, whitewine_df])
    # Quality check
    quality_check(wine_df)
    # Remove wines with unvalid quality scores
    wine_df = wine_df[wine_df.iloc[:, -1].between(0, 10, inclusive='both')]

    return wine_df

def load_houses_dataset():
    """
    Load the Boston house prices dataset from "downloads/housing.data".

    Return:
    df : pandas.DataFrame
        Boston house prices loaded data
    """
    # Read the two type of wine datasets
    houses_df = read_data(pd.read_csv, houses_file, sep='\s+', header=None)
    # Quality check
    quality_check(houses_df)

    return houses_df

def load_dataset(name='wine'):
    """
    Load the dataset given by its name 'wine' or 'houses'.

    Return:
    df : pandas.DataFrame
        Loaded dataset.
    """
    if name == 'wine':
        return load_wine_dataset()
    elif name == 'houses':
        return load_houses_dataset()
    else:
        print("Dataset name unknown.")
        sys.exit()

def quality_check(data):
    """
    Perform quality checks on the given pd.DataFrame.

    Parameters:
    data: pd.DataFrame
        Dataset in pandas format to check.
    """
    # Remove rows with missing values
    data.dropna()
    # Remove duplicated rows
    data.drop_duplicates(subset=None,inplace=True)
    
if __name__ == '__main__':
    # Read datasets
    redwine_df = read_data(pd.read_csv, redwine_file, sep=';')
    whitewine_df = read_data(pd.read_csv, whitewine_file, sep=';')
    house_df = read_data(pd.read_csv, houses_file, sep='\s+', header=None)

    # Concatenate the two type of wine datasets
    wine_df = pd.concat([redwine_df, whitewine_df])

    quality_check(wine_df)
    assert wine_df.isnull().sum().sum() == 0, "Some values are missing in the dataset"
    quality_check(house_df)

    if VERBOSE:
        print("Missing values and duplicates dropped.")
        wine_df.info()
        house_df.info()

    # Remove wines with unvalid quality scores
    wine_df = wine_df[wine_df.iloc[:, -1].between(0, 10, inclusive='both')]
