import os
import requests
from requests.exceptions import RequestException

download_dir = "downloads"

url_red_wine = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'wine-quality/winequality-red.csv')
url_white_wine = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                  'wine-quality/winequality-white.csv')
url_house_prices = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'housing/housing.data')
urls = [url_red_wine, url_white_wine, url_house_prices]

def download_url(url, directory):
    """
    Download file from input url in the given directory folder.
    The file is name after its name contained at the end of the url.

    Parameters:
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
        print(e)

if __name__ == '__main__':
    print('Download script starting...')

    for url in urls:
        download_url(url, download_dir)

    print("All the datasets has been downloaded in the {0}/ folder"
            .format(download_dir))