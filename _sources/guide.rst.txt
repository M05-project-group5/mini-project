.. vim: set fileencoding=utf-8

.. _doc_guide:

==========
User Guide
==========

This package allows you to run a machine learning model on a given dataset with
several parameters that can be tuned.

The current options available for the pipeline can be seen with the help function:

.. code-block:: sh

    $ mini-project-main -h
   
::

    usage: main.py [-h] [-d {wine,houses}] [--seed SEED] [--scaling {min-max,z-normalisation}] [--polynomial]
                    [-m {linear-regression,regression-trees}] [--metrics {mae}]

    Analyze datasets with ML regression techniques.

    optional arguments:
    -h, --help            show this help message and exit
    -d {wine,houses}, --dataset {wine,houses}
                            Dataset to use between wine quality and Boston house prices datasets.
    --seed SEED           Seed for the pseudo-RNG used to split the data and to initialize the models. If no seed is given
                            by the user, the system will be fully random.
    --scaling {min-max,z-normalisation}
                            Select the scaling pre-processing technique to apply to the features.
    --polynomial          Use polynomial features instead of orginial ones for pre-processing
    -m {linear-regression,regression-trees}, --model {linear-regression,regression-trees}
                            Select the ML model that will be used to analyze the data.
    --metrics {mae}       Choose the metrics used as a measure of success of the chosen model.


Default pipeline
----------------

To run the pipeline with the default arguments, run:

.. code-block:: sh

    $ mini-project-main

In this case you are performing linear regression on the Wine quality dataset. 
If you fix the seed to 0 to be reproducible you should have the following output:

::

    dataset    -> wine
    seed       -> 0
    scaling    -> min-max
    polynomial -> None
    model      -> linear-regression
    depth      -> None
    metrics    -> maep
    ===============================
    On the train set: 
    mae = 0.09332051454292202
    On the test set: 
    mae = 0.09469832898909902


Custom pipeline
---------------

Using command-line arguments you can select different parameters or building 
blocks of the pipeline. To check the available options look at the helper function.

For example, to run a regression trees model on the Boston house prices dataset 
with a given seed, run:

.. code-block:: sh

    $ mini-project-main -d houses -m regression-trees --seed 0

An example of a fully customized pipeline is:

.. code-block:: sh

    $ mini-project-main -d wine -m regression-trees --seed 0 --depth 10 --scaling min-max --polynomial 3 --metrics mae


.. include:: links.rst