Contribution via pull requests are always welcome. Source code is available from
`Github`_. Before submitting a pull request, please open an issue to discuss
your changes. Use only the `main` branch as the target branch when submitting a pull request (PR).

.. _`Github` : https://github.com/metatensor/metatomic

Interactions with the metatomic projects must follow our `code of conduct`_.

.. _code of conduct: https://github.com/metatensor/metatomic/blob/main/CODE_OF_CONDUCT.md

Required tools
--------------

You will need to install and get familiar with the following tools when working
on metatomic:

- **git**: the software we use for version control of the source code. See
  https://git-scm.com/downloads for installation instructions.
- **Python**: you can install ``Python`` and ``pip`` on your operating system.
  We require a Python version of at least 3.9.
- **tox**: a Python test runner, see https://tox.readthedocs.io/en/latest/. You
  can install tox with ``pip install tox``.

Additionally, you will need to install the following software, but you should
not have to interact with them directly:

- **cmake**: we need a cmake version of at least 3.16.
- **a C++ compiler** we need a compiler supporting C++11. GCC >= 7, clang >= 5
  and MSVC >= 19 should all work, although MSVC is not yet tested continuously.

.. admonition:: Optional tools

  Depending on which part of the code you are working on, you might experience a
  lot of time spent re-compiling code, even if you did not directly change them.
  For faster builds (and in turn faster tests), you can use compiler cache, like
  `sccache`_ or the classic `ccache`_ to reduce the recompilation of unchanged
  source code. To do this, you should install and configure one of these tools
  (we suggest ``sccache`` since it also supports Rust), and then configure
  ``cmake`` and ``cargo`` to use them by setting environnement variables. On
  Linux and macOS, you should set the following (look up how to do set
  environment variable with your shell):

  .. code-block:: bash

      CMAKE_C_COMPILER_LAUNCHER=sccache
      CMAKE_CXX_COMPILER_LAUNCHER=sccache
      # only if you have sccache and not ccache
      RUSTC_WRAPPER=sccache


  .. _sccache: https://github.com/mozilla/sccache
  .. _ccache: https://ccache.dev/

Getting the code
----------------

The first step when developing ``metatomic`` is to `create a fork`_ of the main
repository on github, and then clone it locally:

.. code-block:: bash

    git clone <insert/your/fork/url/here>
    cd metatomic

    # setup the local repository so that the main branch tracks changes in
    # the original repository
    git remote add upstream https://github.com/metatensor/metatomic
    git fetch upstream
    git branch main --set-upstream-to=upstream/main

Once you get the code locally, you will want to run the tests to check
everything is working as intended. See the next section on this subject.

If everything is working, you can create your own branches to work on your
changes:

.. code-block:: bash

    git checkout -b <my-branch-name>
    # code code code

    # push your branch to your fork
    git push -u origin <my-branch-name>
    # follow the link in the message to open a pull request (PR)

.. _create a fork: https://docs.github.com/en/github/getting-started-with-github/fork-a-repo

Running tests
-------------

The continuous integration pipeline is based on `tox`_. You can run all tests
with:

.. code-block:: bash

    cd <path/to/metatomic/repo>
    tox

These are exactly the same tests that will be performed online in our Github CI
workflows. You can also run only a subset of tests with one of these commands:

.. code-block:: bash

    tox -e lint                           # check files for formatting errors

    tox -e torch-tests                    # unit tests for metatomic-torch, in Python
    tox -e torch-tests-cxx                # unit tests for metatomic-torch, in C++
    tox -e torch-install-tests-cxx        # testing that the C++ code is a valid CMake package
    tox -e docs-tests                     # doctests (checking inline examples) for all packages
    tox -e lint                           # code style

    tox -e format                         # format all files

The last command ``tox -e format`` will use ``tox`` to do actual formatting
instead of just checking it, you can use this to automatically fix some of the
issues detected by ``tox -e lint``.

You can run only a subset of the tests with ``tox -e torch-tests --
<test/file.py>``, replacing ``<test/file.py>`` with the path to the files you
want to test, e.g. ``tox -e tests -- tests/system.py``.

Controlling test behavior with environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are a handful of environment variables that you can set to control the
behavior of tests:

- ``METATOMIC_DISABLE_VALGRIND=1`` will disable the use of `valgrind`_ for the
  C++ tests. Valgrind is a tool that check for memory errors in native code, but it makes the tests run quite a bit slower;
- ``METATOMIC_TESTS_TORCH_VERSION`` allow you to run the tests against a
  specific PyTorch version instead of the latest one. For example, setting
  ``METATOMIC_TESTS_TORCH_VERSION=2.4`` will run the tests against PyTorch
  2.4;
- ``PIP_EXTRA_INDEX_URL`` can be used to pull PyTorch (or other dependencies)
  from a different index. This can be useful on Linux if you have issues with
  CUDA, since the default PyTorch version expects CUDA to be available. A
  possible workaround is to use the CPU-only version of PyTorch in the tests, by
  setting ``PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu``;
- ``PYTORCH_JIT=0`` can be used to disable Python to TorchScript compilation of
  code; producing error messages which should be easier to understand.

.. _valgrind: https://valgrind.org/

Inspecting Python code coverage
-------------------------------

The code coverage is reported at `codecov`_. You can also inspect the coverage
locally. Python coverage is written out as several individual files. It is
easier to combine all reports and open the generated ``html`` file in a browser

.. code-block:: bash

    tox
    coverage combine .tox/*/.coverage
    coverage html
    firefox htmlcov/index.html

.. _codecov: https://codecov.io/gh/metatensor/metatomic

Contributing to the documentation
---------------------------------

The documentation of ``metatomic`` is written in reStructuredText (rst) and uses
the `sphinx`_ documentation generator. In order to modify the documentation,
first create a local version of the code on your machine as described above.
Then, you can build the documentation with:

.. code-block:: bash

    tox -e docs

In addition to the requirements listed above, you will also need to install
`doxygen`_ (e.g. ``apt install doxygen`` on Debian-based systems).

You can then visualize the local documentation with your favorite browser with
the following command (or open the :file:`docs/build/html/index.html` file
manually).

.. code-block:: bash

    # on linux, depending on what package you have installed:
    xdg-open docs/build/html/index.html
    firefox docs/build/html/index.html

    # on macOS:
    open docs/build/html/index.html

It may be easier to run a web-server to ensure links resolve correctly. For
that:

.. code-block:: bash

    # on linux, depending on what package you have installed:
    python -m http.server -d docs/build/html
    # Go to localhost:8080 on the browser

.. _`sphinx` : https://www.sphinx-doc.org/en/master/
.. _`doxygen` : https://www.doxygen.nl/index.html

Python doc strings
~~~~~~~~~~~~~~~~~~

Our docstring format follows the `sphinx format`_ guidelines and a typical
doc string for a function looks like the following.

.. code-block:: python

    def func(value_1: float, value_2: int) -> float:
        r"""A one line summary sentence of the function.

        Extensive multi-line summary of what is going in. Use single backticks
        for parameters of the function like `width` and two ticks for values
        ``67``. You can link to classes :py:class:`metatomic.torch.System`. This
        also works for other classes and functions like
        :py:class:`torch.Tensor`.

        Inline Math is also possible with :math:`\mathsf{R}`. Or as a math block.

        .. math::

            \mathbf{x}' = \mathsf{R}\mathbf{x}


        :param value_1:
            The first parameter of the function, a :py:class:`float`.
        :param value_2:
            The second parameter of the function, an :py:class:`int`.

        :returns result:
            The result of the calculation, a :py:class:`float`.

        :raises TypeError:
            If `value_1` is not a :py:class:`float` or `value_2` is not a :py:class:`int`.
        :raises ValueError:
            If `value_1` is not greater than zero.

        Examples
        --------
        >>> from metatomic import func
        >>> func(1, 1)
        42
        """
        ...
        return result

Guidelines for writing Python doc strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use Python typing in the function arguments, and indicate return types.

* Start the description after each ``:param:`` or ``:return:`` in a new line and add an
  empty line between the parameter and return block.

* Emphasize function and class parameters with a single backtick i.e ```param``` and
  general variables with a double backtick . i.e. ````my_variable````

* If you include any mathematical formulae, use a `raw string`_ by prefixing the
  string with ``r``, e.g.,

  .. code-block:: python

    r"""Some math like :math:`\nu^2 / \rho` with backslashes."""

  Otherwise, the ``\n`` and ``\r`` will be rendered as ASCII escape sequences
  that break lines without you noticing it or you will get either one of the
  following two errors message

  1. `Explicit markup ends without a blank line; unexpected unindent`
  2. `Inline interpreted text or phrase reference start-string without end string`

* The examples are tested with `doctest`_. Therefore, please make sure that they are
  complete and functioning (with all required imports).
  Use the ``>>>`` syntax for inputs (followed by ``...`` for multiline inputs) and no
  indentation for outputs for the examples.

  .. code-block:: python

      """
      >>> a = np.array(
      ...    [1, 2, 3, 4]
      ... )
      """

.. _`sphinx format` : https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
.. _`raw string` : https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals
.. _`doctest` : https://docs.python.org/3/library/doctest.html

Useful developer scripts
------------------------

The following script can be useful to contributors:

- ``./scripts/clean-python.sh``: remove all generated files related to Python,
  including all build caches

Additional scripts, including some release helpers are in the ``scripts/`` folder.
