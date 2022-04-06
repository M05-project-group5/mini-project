.. vim: set fileencoding=utf-8 :

.. _doc_troubleshooting:

Troubleshooting
---------------

For run the unit test we use pytest_:


.. code-block:: shell

  # use your package manager to install the package "pytest"
  # here, I examplify with "pip":
  $ pip install pytest
  $ pytest
  ....
  ---------------------------------------------------------------------
  ======================== test session starts ========================
  ...
  ============ 40 passed, 2 skipped, 13 warnings in 8.66s =============

If you will know the coverage of test, you can use this command:

.. code-block:: shell

  $ pytest --cov=.

  ======================== test session starts ========================
  ...
  ---------- coverage: platform linux, python 3.8.8-final-0 -----------
  Name                          Stmts   Miss  Cover
  -------------------------------------------------
  {files name}                      *      *     *%
  -------------------------------------------------
  TOTAL                             *      *     *%
  ============ 40 passed, 2 skipped, 13 warnings in 8.66s =============
  
But you also can see this result in COVERALLS_.

.. include:: links.rst