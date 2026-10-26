Model wrappers
==============

These classes are used to wrap existing models and provide additional
functionalities, such as new outputs or modified behavior. The following
wrappers are available:

- :py:class:`metatomic_torch.HeatFlux` adds :ref:`heat flux
  <heat-flux-quantity>` output(s) to a model based on its existing :ref:`energy
  <energy-quantity>` output(s).
- :py:class:`metatomic_torch.DFTD3` add DFT-D3 dispersion energy to existing
  :ref:`energy <energy-quantity>` output(s).
- :py:class:`metatomic_torch.WeightedSum` combines multiple :ref:`variants
  <quantity-variants>` of an output with fixed weights.

--------------------------------------------------------------------------------

.. autoclass:: metatomic_torch.HeatFlux
    :members:
    :exclude-members: forward


.. autoclass:: metatomic_torch.DFTD3
    :members:
    :exclude-members: forward


.. autoclass:: metatomic_torch.WeightedSum
    :members:
    :exclude-members: forward
