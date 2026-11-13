.. _torch-api:

PyTorch API reference
=====================

The PyTorch API of metatomic (in the ``metatomic-torch`` package) defines
similar classes as the :ref:`Python API <python-api-core>`, the main difference
being that the classes in the ``metatomic_torch`` module can be used together
with TorchScript (and ``torch.compile`` in the future).

Classes from the legacy (v0.1) API are also included for backward compatibility,
but should not be used for new code and will be made hidden in the future.

.. toctree::
    :maxdepth: 1

    system
    metadata
    run-model
    new-model
    misc

    legacy

    wrappers
    o3/index

.. toctree::
    :maxdepth: 1
    :hidden:

    CHANGELOG.md
