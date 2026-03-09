Miscellaneous
=============

.. autofunction:: metatomic.torch.pick_device

.. autofunction:: metatomic.torch.pick_output

.. autofunction:: metatomic.torch.unit_conversion_factor

.. _known-quantities-units:

Known quantities
----------------

When setting ``quantity`` on a :py:class:`~metatomic.torch.ModelOutput`, the
following names are recognized. The parser will check that the unit expression
has dimensions matching the expected quantity.

.. list-table:: Physical Dimensions
   :header-rows: 1

   * - quantity
     - expected dimension
   * - **length**
     - :math:`L`
   * - **energy**
     - :math:`M L^2 T^{-2}`
   * - **force**
     - :math:`M L T^{-2}`
   * - **pressure**
     - :math:`M L^{-1} T^{-2}`
   * - **momentum**
     - :math:`M L T^{-1}`
   * - **mass**
     - :math:`M`
   * - **velocity**
     - :math:`L T^{-1}`
   * - **charge**
     - :math:`Q`

.. note::

   The 3-argument form ``unit_conversion_factor(quantity, from_unit, to_unit)``
   is deprecated. Use the 2-argument form instead. The ``quantity`` parameter is
   ignored by the parser; dimensional compatibility is checked automatically.
