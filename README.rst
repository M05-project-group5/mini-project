.. image:: https://github.com/M05-project-group5/mini-project/actions/workflows/ci-testing.yml/badge.svg?branch=main
   :target: https://github.com/M05-project-group5/mini-project/actions/workflows/ci-testing.yml
.. image:: https://coveralls.io/repos/github/M05-project-group5/mini-project/badge.svg?branch=main
   :target: https://coveralls.io/github/M05-project-group5/mini-project?branch=main
.. image:: https://img.shields.io/badge/docs-latest-orange.svg
   :target: https://M05-project-group5.github.io/mini-project/index.html

============
Mini-project
============

This is an extensible and fully reproducible system to analyze multiple datasets, with various Machine Learning techniques.

Description
===========

More information on the project.

Getting Started
===============

Installation
------------

1. Clone the repo ::

    git clone https://github.com/M05-project-group5/mini-project.git

2. Install dependencies in a conda env::
   
    conda env create -f environment.yml -n mini-project
   
Usage
-----

1. Activate the environment::

    conda activate mini-project

2. Run the pipeline with default arguments::

    cd mini-project
    python main.py

Download 'manually' the datasets
--------------------------------

   python download_datasets.py
   
Choose blocks of the pipeline
=============================

Run the main script with command-line arguments to select and change different elements of the pipeline.
Below is the help informations provided for the main script::

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

  
For example, to train a Regression trees model on the Wine quality dataset with a z-normalisation, run::

   python main.py -d wine --scaling z-normalisation -m regression-trees
 
Authors
=======
Cédric Mariéthoz \
Adrien Chassignet
