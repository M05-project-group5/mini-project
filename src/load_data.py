import pandas as pd

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

def load_wine_dataset(whitewine_file="downloads/winequality-white.csv",
                     redwine_file="downloads/winequality-red.csv"):
    """
    Load the wine quality dataset from given files. Both dataset are merged.

    Parameters:
    whitewine_file : string
        Relative path to the file containing the white wine quality dataset.
        Default is "downloads/winequality-white.csv".
    redwine_file : string
        Relative path to the file containing the red wine quality dataset
        Default is "downloads/winequality-red.csv".

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

def load_houses_dataset(houses_file="downloads/housing.data"):
    """
    Load the Boston house prices dataset from the given file.
    
    Parameters:
    houses_file : string
        Relative path to the file containing the Boston house prices dataset.
        Default is "downloads/housing.data".
    Return:
    df : pandas.DataFrame
        Boston house prices loaded data
    """
    # Read the two type of wine datasets
    houses_df = read_data(pd.read_csv, houses_file, sep=r'\s+', header=None)
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
        raise SystemExit("Dataset name unknown.")

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
