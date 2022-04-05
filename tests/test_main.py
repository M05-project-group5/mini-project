import pytest
import sys
sys.path.insert(1, './src')

import os
import main

@pytest.mark.dependency(name="default")
def test_get_cl_args_default():
    """ Test if default call of the argument parser is correct. """
    args = main.get_cl_args(args=[])
    assert args.dataset, "Dataset not set."
    assert args.seed == None, "Seed should be None by default."
    assert args.scaling, "Scaling preprocessing not set."
    assert args.polynomial == None, "Polynomes should not be used by default."
    assert args.model, "Model not set."
    assert args.depth == None, "No max_depth should be set by default."
    assert args.metrics, "Metric not set."

def test_get_cl_args_wrong_dataset():
    """ Call the argument parser with an unsupported dataset. """
    with pytest.raises(SystemExit) as err:
        main.get_cl_args(["-d dummy"])
    assert err.value.code == 2, "Program should exit with a code 2 error."

def test_get_cl_args_wrong_preprocessing():
    """ Call the argument parser with an unsupported preprocessing. """
    with pytest.raises(SystemExit) as err:
        main.get_cl_args(["--scaling dummy"])
    assert err.value.code == 2, "Program should exit with a code 2 error."

def test_get_cl_args_wrong_model():
    """ Call the argument parser with an unsupported model. """
    with pytest.raises(SystemExit) as err:
        main.get_cl_args(["-m dummy"])
    assert err.value.code == 2, "Program should exit with a code 2 error."

def test_get_cl_args_wrong_metric():
    """ Call the argument parser with an unsupported figure of merit. """
    with pytest.raises(SystemExit) as err:
        main.get_cl_args(["--metrics dummy"])
    assert err.value.code == 2, "Program should exit with a code 2 error."

def test_get_cl_args_wrong_argument():
    """ Call the argument parser with an unknown argument. """
    with pytest.raises(SystemExit) as err:
        main.get_cl_args(["--dummy"])
    assert err.value.code == 2, "Program should exit with a code 2 error."

@pytest.mark.dependency(depends=["default"])
@pytest.mark.slow
def test_main_wine_dataset_not_found():
    """ Test that the pipeline download the wine datasets if unavailable. """
    path_redwine = "downloads/winequality-red.csv"
    path_whitewine = "downloads/winequality-white.csv"
    try:
        os.remove(path_redwine)
    except OSError:
        pass
    try:
        os.remove(path_whitewine)
    except OSError:
        pass

    args = main.get_cl_args(args=[])
    args.dataset = 'wine'
    main.main(args)
    assert os.path.exists(path_redwine), "Red Wine Quality dataset not downloaded."
    assert os.path.exists(path_whitewine), "White Wine Quality dataset not downloaded."

@pytest.mark.dependency(depends=["default"])
@pytest.mark.slow
def test_main_houses_dataset_not_found():
    """ Test that the pipeline download the houses dataset if unavailable. """
    path_houses = "downloads/housing.data"
    try:
        os.remove(path_houses)
    except OSError:
        pass

    args = main.get_cl_args(args=[])
    args.dataset = 'houses'
    main.main(args)
    assert os.path.exists(path_houses), "Boston house prices dataset not downloaded."

@pytest.mark.dependency(depends=["default"])
def test_main_wrong_dataset():
    """ Test main with unknown dataset argument. """
    args = main.get_cl_args(args=[])
    args.dataset = 'dummy'
    with pytest.raises(Exception) as exc:
        main.main(args)

@pytest.mark.dependency(depends=["default"])
def test_main_float_seed():
    """ Test main with float seed argument. """
    args = main.get_cl_args(args=[])
    args.seed = 0.4
    with pytest.raises(Exception) as exc:
        main.main(args)

@pytest.mark.dependency(depends=["default"])
def test_main_wrong_scaling():
    """ Test main with unknown scaling argument. """
    args = main.get_cl_args(args=[])
    args.scaling = 'dummy'
    with pytest.raises(Exception) as exc:
        main.main(args)

@pytest.mark.skip(reason="Polynomial argument will be update in a future enhancement.")
@pytest.mark.dependency(depends=["default"])
def test_main_wrong_polynomial():
    """ Test main with unknown polynomial argument. """
    pass

@pytest.mark.dependency(depends=["default"])
def test_main_wrong_model():
    """ Test main with unknown model argument. """
    args = main.get_cl_args(args=[])
    args.model = 'dummy'
    with pytest.raises(Exception) as exc:
        main.main(args)

@pytest.mark.dependency(depends=["default"])
def test_main_wrong_metrics():
    """ Test main with unknown metrics argument. """
    args = main.get_cl_args(args=[])
    args.metrics = 'dummy'
    with pytest.raises(Exception) as exc:
        main.main(args)
