#!/usr/bin/env python3
"""
This script downloads the Wine quality and Boston house prices datasets.

The datasets are downloaded in the downloads/ folder.
Both datasets are downloaded from the UCI Machine Learning Repositiory.
"""
#Author:      Adrien Chassignet
#Co-authir:   Cédric Mariéthoz
#Date:        Feb 28 2022
#Change date: Mar 3 2022 
#Version:     1.1
#Links:       https://archive.ics.uci.edu/ml/datasets/wine+quality
#             https://archive.ics.uci.edu/ml/machine-learning-databases/housing/


import os
import requests
from requests.exceptions import RequestException

download_dir = "../downloads"

url_red_wine = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'wine-quality/winequality-red.csv')
url_white_wine = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                  'wine-quality/winequality-white.csv')
url_house_prices = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'housing/housing.data')
urls = [url_red_wine, url_white_wine, url_house_prices]

def download_wine():
    """Download the 2 wine quality datasets in the downloads/ folder if the files
    do not exist.
    """
    if ((not os.path.isfile(download_dir + '/' + url_red_wine.split("/")[-1])) or
            (not os.path.isfile(download_dir + '/' + url_white_wine.split("/")[-1]))):
        download_url(url_red_wine, download_dir)
        download_url(url_white_wine, download_dir)
        print("Wine datasets downloaded.")

def download_houses():
    """Download the Boston house prices dataset in the downloads/ folder if the
    file does not exists.
    """
    if not os.path.isfile(download_dir + '/' + url_house_prices.split("/")[-1]):
        download_url(url_house_prices, download_dir)
        print("Boston house prices dataset downloaded.")

def download_url(url, directory):
    """Download file from input url in the given directory folder.

    The file is named after its name contained at the end of the url.

    Parameters
    ----------
    url : string
        Url to download the data.
    directory : string
        Name of the directory where the downloaded files will be saved
    """
    try:
        print("Downloading from " + url)
        r = requests.get(url)

        if not os.path.isdir(directory):
            os.makedirs(directory)
            print(directory + " folder has been created.")

        with open(directory + '/' + url.split("/")[-1], 'wb') as f:
            f.write(r.content)
    except RequestException as e:
        raise SystemExit(e)

if __name__ == '__main__':  # pragma: no cover
    print('Download script starting...')

    for url in urls:
        download_url(url, download_dir)

    print("All the datasets has been downloaded in the {0}/ folder"
            .format(download_dir))