import numpy as np
import pandas as pd

VERBOSE = True

# Read datasets
redwine_df = pd.read_csv("downloads/winequality-red.csv", sep=';')
whitewine_df = pd.read_csv("downloads/winequality-white.csv", sep=';')

# Concatenate the two type of wine datasets
wine_df = pd.concat([redwine_df, whitewine_df])

if VERBOSE:
    wine_df.info()
    
# Remove rows with missing values
wine_df.dropna()
assert wine_df.isnull().sum().sum() == 0, "Some values are missing in the dataset"

# Remove duplicated rows
wine_df.drop_duplicates(subset=None,inplace=True)

if VERBOSE:
    print("Duplicates dropped.")
    wine_df.info()

# Remove wines with unvalid quality scores
wine_df = wine_df[wine_df.iloc[:, -1].between(0, 10, inclusive=True)]

if VERBOSE:
    print("Quality scores checked.")
    wine_df.info()
