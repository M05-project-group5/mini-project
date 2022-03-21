import pytest
import sys
sys.path.insert(1, './src')
sys.path.insert(1, '.')

import load_data
import download_datasets

def test_download_wine():
    """ Test if wine quality datasets are correctly downloaded. """

def test_load_dataset_incorrect_name():
    """ Test if system exits correctly when dataset name is not recognized. """
    with pytest.raises(SystemExit) as e:
        load_data.load_dataset(name='dummy')