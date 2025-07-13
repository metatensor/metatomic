.. _installation:

Installation
============

Metatomic is available for multiple programming languages, and how to install
and use it will depend on the programming language you are using.

.. tab-set::

    .. tab-item:: TorchScript Python
        :name: install-torch

        The TorchScript bindings to metatomic are accessible in Python in the
        ``metatomic-torch`` package. You can install them with

        .. code-block:: bash

            # Make sure you are using the latest version of pip
            pip install --upgrade pip

            pip install metatomic-torch

        We provide pre-compiled wheels on PyPI that are compatible with all the
        supported ``torch`` versions at the time of the ``metatomic-torch`` release.
        Currently PyTorch version 2.1 and above is supported.

        If you want to use the code with an unsupported PyTorch version, or a
        new release of PyTorch which did not exist yet when we released
        ``metatomic-torch``; you'll need to compile the code on your local machine
        with

        .. code-block:: bash

            pip install metatomic-torch --no-binary=metatomic-torch

        This local compilation will require a couple of additional dependencies:

        - a modern C++ compiler, able to handle C++17, such as:
            - gcc version 7 or above;
            - clang version 5 or above;
            - Microsoft Visual C++ (MSVC) compiler, version 19 (2015) or above.
        - if you want to use the CUDA version of PyTorch, you'll also need the
          `CUDA toolkit <https://developer.nvidia.com/cuda-toolkit>`_, including
          the NVIDIA compiler.

        By default, PyTorch is installed with CUDA support on Linux, even if you
        do not have a CUDA-compatible GPU, and will search for a CUDA toolkit
        when building extension (such as ``metatomic-torch``). If you don't want
        to install the CUDA toolkit in this case, you can use the CPU-only
        version of PyTorch with

        .. code-block:: bash

            pip install --extra-index-url https://download.pytorch.org/whl/cpu metatomic-torch --no-binary=metatomic-torch

        A similar index URL can be used to install the ROCm (AMD GPU) version of
        PyTorch, please refer to the `corresponding documentation
        <https://pytorch.org/get-started/locally/>`_.

        .. seealso::

            Some potential build failures and corresponding workarounds are
            listed at the end of the :ref:`install-torch-cxx` installation
            instructions.


    .. tab-item:: TorchScript C++
        :name: install-torch-cxx

        The TorchScript bindings to metatomic are also available as a C++
        library, which can be integrated in non-Python software (such as
        simulation engines) to use custom metatomic models directly in the
        software without relying on a Python interpreter. The code is installed
        as a shared library which register itself with torch when loaded, with
        the corresponding header files and a CMake integration allowing you to
        use metatomic-torch in your code code with
        ``find_package(metatomic_torch)``.

        To build and install the code, you'll need to find the latest release of
        ``metatomic-torch`` on `GitHub releases
        <https://github.com/metatensor/metatomic/releases>`_, and download the
        corresponding ``metatomic-torch-cxx`` file in the release assets. Then,
        you can run the following commands:

        .. code-block:: bash

            cmake -E tar xf metatomic-torch-cxx-*.tar.gz
            cd metatomic-torch-cxx-*
            mkdir build && cd build

            # configure cmake here if needed
            cmake ..

            # build and install the code
            cmake --build . --target install

        You will have to to manually install some of the dependencies of
        ``metatomic-torch`` yourself to compile this code, and if any of the
        dependencies is not in a standard location, specify the installation
        directory when configuring cmake with ``CMAKE_PREFIX_PATH``. The
        following dependencies might have to be installed beforehand:

        - :external+metatensor:ref:`the C++ interface <install-torch-cxx>` of
          metatensor-torch.
        - the C++ part of PyTorch, which you can install `on its own
          <https://pytorch.org/get-started/locally/>`_. We are compatible with
          ``libtorch`` version 2.1 or above. You can also use the same library as
          the Python version of torch by adding the output of the command below
          to ``CMAKE_PREFIX_PATH``:

          .. code-block:: bash

              python -c "import torch; print(torch.utils.cmake_prefix_path)"


        +--------------------------------------+-----------------------------------------------+----------------+
        | Option                               | Description                                   | Default        |
        +======================================+===============================================+================+
        | ``CMAKE_BUILD_TYPE``                 | Type of build: debug or release               | release        |
        +--------------------------------------+-----------------------------------------------+----------------+
        | ``CMAKE_INSTALL_PREFIX``             | Prefix in which the library will be installed | ``/usr/local`` |
        +--------------------------------------+-----------------------------------------------+----------------+
        | ``CMAKE_PREFIX_PATH``                | ``;``-separated list of path where CMake will |                |
        |                                      | search for dependencies. This list should     |                |
        |                                      | include the path to metatensor and torch      |                |
        +--------------------------------------+-----------------------------------------------+----------------+

        **Workaround for some build errors**

        The CMake configuration used by ``libtorch`` sometimes fails to setup the
        build environment. You'll find here a list of some known build failures
        and how to workaround them.

        - .. code-block:: text

              Unknown CUDA Architecture Name 9.0a in CUDA_SELECT_NVCC_ARCH_FLAGS

          This can happen when building with a CUDA-enabled version of ``torch`` and
          a recent version of ``cmake``. This issue is tracked at
          https://github.com/pytorch/pytorch/issues/113948. To work around it,
          you can ``export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"`` in your
          environment before building the code.

        - .. code-block:: text

              Imported target "torch" includes non-existent path
                [...]/MKL_INCLUDE_DIR-NOTFOUND"
              in its INTERFACE_INCLUDE_DIRECTORIES.

          This can happen when building for x86_64 Linux when the `Intel-MKL`_
          is not available on the current machine. Since MKL is a completely
          optional dependency, you can silence the error by running cmake with
          the ``-DMKL_INCLUDE_DIR=/usr/include`` option.

          .. _Intel-MKL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html



Installing a development version
--------------------------------

Metatomic is developed on `GitHub <https://github.com/metatensor/metatomic>`_.
If you want to install a development version of the code, you will need `git
<https://git-scm.com>`_ to fetch the latest version of the code.


.. tab-set::

    .. tab-item:: TorchScript Python
        :name: dev-install-torch

        You can install a development version of the TorchScript bindings to
        ``metatomic`` with:

        .. code-block:: bash

            # Make sure you are using the latest version of pip
            pip install --upgrade pip

            git clone https://github.com/metatensor/metatomic
            cd metatomic
            pip install ./python/metatomic_torch

            # alternatively, the same thing in a single command
            pip install "metatomic-torch @ git+https://github.com/metatensor/metatomic#subdirectory=python/metatomic_torch"


    .. tab-item:: TorchScript C++
        :name: dev-install-torch-cxx

        You can install the development version of ``metatomic`` with the following
        (the same :ref:`cmake configuration options <install-torch-cxx>` are
        available):

        .. code-block:: bash

            git clone https://github.com/metatensor/metatomic
            cd metatomic/metatomic-torch
            mkdir build && cd build

            # configure cmake here if needed
            cmake ..

            # build and install the code
            cmake --build . --target install



.. _pip: https://pip.pypa.io
.. _CMake: https://cmake.org
