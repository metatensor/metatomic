Model wrappers
==============

These classes are used to wrap existing models and provide additional
functionalities, such as new outputs or modified behavior.

.. autoclass:: metatomic.torch.heat_flux.HeatFlux
    :members:
    :exclude-members: forward


.. autoclass:: metatomic.torch.dftd3.DFTD3
    :members:
    :exclude-members: forward


.. autoclass:: metatomic.torch.weighted_sum.WeightedSum
    :members:
    :exclude-members: forward


Running models with ``metatomic``
---------------------------------

:py:class:`LegacyModel <metatomic.torch.legacy.LegacyModel>` wraps an
:py:class:`AtomisticModel <metatomic.torch.AtomisticModel>` into a
:py:class:`metatomic.TorchModel`, so existing models (including ``.pt`` files)
can run with :py:func:`metatomic.execute_model`.

.. autoclass:: metatomic.torch.legacy.LegacyModel
    :members: load
    :exclude-members: forward

.. autofunction:: metatomic.torch.legacy.system_with_pairs
