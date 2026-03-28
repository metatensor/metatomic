.. _atomistic-models-inputs:

Standard model inputs
=====================

Some simulation engines can provide additional per-system inputs to a model,
beyond the atomic positions and species. If your model expects one of the
inputs defined in this documentation, it should use the corresponding
standardized name and follow the metadata structure described here.

If you need other inputs, you should use a custom name containing ``::``,
such as ``my_code::my_input``.

.. toctree::
  :maxdepth: 1
  :hidden:

  charge
  spin

Physical quantities
^^^^^^^^^^^^^^^^^^^

.. grid:: 1 2 2 2

    .. grid-item-card:: Charge
      :link: charge-input
      :link-type: ref

      The total electric charge of the system, in units of the elementary
      charge :math:`e`.

    .. grid-item-card:: Spin
      :link: spin-input
      :link-type: ref

      The spin multiplicity :math:`2S + 1` of the system.
