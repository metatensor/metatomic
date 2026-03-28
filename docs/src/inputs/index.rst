.. _atomistic-models-inputs:

Standard model inputs
=====================

Models can receive additional per-system inputs beyond the atomic positions and
species. These inputs must be set by the user (e.g. via ``atoms.info`` in ASE)
before running a calculation. If your model expects one of the inputs defined
in this documentation, it should use the corresponding standardized name and
follow the metadata structure described here.

If you need other inputs, you should use a custom name containing ``::``,
such as ``my_code::my_input``.

.. toctree::
  :maxdepth: 1
  :hidden:

  charge
  spin

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
