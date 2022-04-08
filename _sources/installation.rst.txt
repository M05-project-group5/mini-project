.. vim: set fileencoding=utf-8

.. _doc_installation:

============
Installation
============

Use pip:

.. code-block:: sh

    $ python -m venv ~/venv                # A new env with no packages
    $ source ~/venv/bin/activate
    (venv) $ pip list                      # Control for no packages
    ...
    (venv) $ pip install --extra-index-url https://test.pypi.org/simple Mini-Project
    (venv) $ mini-project-main
    ...

.. include:: links.rst