import pytest
import sys
sys.path.insert(1, './src')
sys.path.insert(1, '.')

import load_data
import download_datasets
import os

def test_download_url_incorrect():
    """ Test if system exits correctly when trying to download from unknown url. """
    with pytest.raises(SystemExit) as e:
        download_datasets.download_url("https://dummy-url.com",
                                        download_datasets.download_dir)

@pytest.mark.slow
def test_download_wine():
    """ Test if wine quality datasets are correctly downloaded. """
    path_redwine = "downloads/winequality-red.csv"
    path_whitewine = "downloads/winequality-white.csv"
    if os.path.exists(path_redwine):
        os.remove(path_redwine)
    if os.path.exists(path_whitewine):
        os.remove(path_whitewine)

    download_datasets.download_wine()
    assert os.path.exists(path_redwine), "Red Wine Quality dataset not downloaded."
    assert os.path.exists(path_whitewine), "White Wine Quality dataset not downloaded."

@pytest.mark.slow
def test_download_houses():
    """ Test if Boston house prices dataset is correctly downloaded. """
    path_houses = "downloads/housing.data"
    if os.path.exists(path_houses):
        os.remove(path_houses)

    download_datasets.download_houses()
    assert os.path.exists(path_houses), "Boston house prices dataset not downloaded."


def test_load_dataset_incorrect_name():
    """ Test if system exits correctly when dataset name is not recognized. """
    with pytest.raises(SystemExit) as e:
        load_data.load_dataset(name='dummy')