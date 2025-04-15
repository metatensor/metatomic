TorchScript backend
===================

.. toctree::
    :maxdepth: 2
    :hidden:

    reference/index

.. toctree::
    :maxdepth: 1
    :hidden:

    CHANGELOG.md

We provide a special PyTorch C++ extension exporting all of the core metatensor
types in a way compatible with the TorchScript compiler, allowing users to save
and load models based on metatensor everywhere TorchScript is supported. This
allow to define, train and save a model from Python, and then load it with pure
C++ code, without requiring a Python interpreter. Please refer to the
:ref:`installation instructions <install-torch>` to know how to install the
Python and C++ sides of this library.

.. grid::

    .. grid-item-card:: |Python-16x16| TorchScript Python API reference
        :link: python-api-torch
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the classes and functions in the
        ``metatensor-torch`` Python package.

        +++
        Documentation for version |metatomic-torch-version|

    .. grid-item-card:: |Cxx-16x16| TorchScript C++ API reference
        :link: cxx-api-torch
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the classes and functions in the
        ``metatensor/torch.hpp`` C++ header.

        +++
        Documentation for version |metatomic-torch-version|
