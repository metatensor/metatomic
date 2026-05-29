Miscellaneous
=============

.. autofunction:: metatomic.torch.pick_device

.. autofunction:: metatomic.torch.pick_output

.. autofunction:: metatomic.torch.unit_conversion_factor

The set of recognized base units is documented in the
:cpp:func:`C++ API reference <metatomic_torch::unit_conversion_factor>`.

The :py:func:`unit_conversion_factor` function accepts any valid unit expression
built from base units combined with operators. There is no need to specify a
physical quantity --- the parser automatically verifies dimensional compatibility
between the source and target units.
