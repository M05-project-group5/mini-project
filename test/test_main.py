import pytest
import sys
from argparse import Namespace
from main import get_cl_args, main

@pytest.mark.dependency(name="default")
def test_get_cl_args_default():
    """ Test if default call of the argument parser is correct. """
    args = get_cl_args(args=[])
    assert args.dataset, "Dataset not set."
    assert args.seed == None, "Seed should be None by default."
    assert args.scaling, "Scaling preprocessing not set."
    assert args.polynomial == False, "Polynomes should not be used by default."
    assert args.model, "Model not set."
    assert args.metrics, "Metric not set."

@pytest.mark.dependency()
def test_get_cl_args_wrong_dataset():
    """ Call the argument parser with an unsupported dataset. """
    with pytest.raises(SystemExit) as err:
        get_cl_args(["-d dummy"])
    assert err.value.code == 2, "Program should exit with a code 2 error."

@pytest.mark.dependency()
def test_get_cl_args_wrong_preprocessing():
    """ Call the argument parser with an unsupported preprocessing. """
    with pytest.raises(SystemExit) as err:
        get_cl_args(["--scaling dummy"])
    assert err.value.code == 2, "Program should exit with a code 2 error."

@pytest.mark.dependency()
def test_get_cl_args_wrong_model():
    """ Call the argument parser with an unsupported model. """
    with pytest.raises(SystemExit) as err:
        get_cl_args(["-m dummy"])
    assert err.value.code == 2, "Program should exit with a code 2 error."

@pytest.mark.dependency()
def test_get_cl_args_wrong_metric():
    """ Call the argument parser with an unsupported figure of merit. """
    with pytest.raises(SystemExit) as err:
        get_cl_args(["--metrics dummy"])
    assert err.value.code == 2, "Program should exit with a code 2 error."

@pytest.mark.dependency()
def test_get_cl_args_wrong_argument():
    """ Call the argument parser with an unknown argument. """
    with pytest.raises(SystemExit) as err:
        get_cl_args(["--dummy"])
    assert err.value.code == 2, "Program should exit with a code 2 error."

@pytest.mark.dependency(depends=["default"])
def test_main_wrong_dataset():
    """ Test main with unknown dataset argument. """
    args = get_cl_args(args=[])
    args.dataset = 'dummy'
    with pytest.raises(Exception) as exc:
        main(args)
    assert "Unknown dataset for this pipeline." in str(exc.value)

@pytest.mark.dependency(depends=["default"])
def test_main_float_seed():
    """ Test main with float seed argument. """
    args = get_cl_args(args=[])
    args.seed = 0.4
    with pytest.raises(Exception) as exc:
        main(args)
    assert '0.4 cannot be used to seed' in str(exc.value)

@pytest.mark.dependency(depends=["default"])
def test_main_wrong_scaling():
    """ Test main with unknown scaling argument. """
    args = get_cl_args(args=[])
    args.scaling = 'dummy'
    with pytest.raises(Exception) as exc:
        main(args)
    assert "Unknown pre-processing for this pipeline." in str(exc.value)

@pytest.mark.skip(reason="Polynomial argument will be update in a future enhancement.")
@pytest.mark.dependency(depends=["default"])
def test_main_wrong_polynomial():
    """ Test main with unknown polynomial argument. """
    assert False

@pytest.mark.dependency(depends=["default"])
def test_main_wrong_model():
    """ Test main with unknown model argument. """
    args = get_cl_args(args=[])
    args.model = 'dummy'
    with pytest.raises(Exception) as exc:
        main(args)
    assert "Unknown model for this pipeline." in str(exc.value)

@pytest.mark.dependency(depends=["default"])
def test_main_wrong_metrics():
    """ Test main with unknown metrics argument. """
    args = get_cl_args(args=[])
    args.metrics = 'dummy'
    with pytest.raises(Exception) as exc:
        main(args)
    assert "Unknown figure of merit for this pipeline." in str(exc.value)