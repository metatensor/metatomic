.. _spin-multiplicity-output:

Spin multiplicity
^^^^^^^^^^^^^^^^^

The spin multiplicity of the system is associated with the
``"spin_multiplicity"`` or ``"spin_multiplicity/<variant>"`` name (see
:ref:`output-variants`), and must have the following metadata:

.. list-table:: Metadata for spin_multiplicity
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The spin multiplicity is always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system"]``
    - the samples must be named ``["system"]``, since the spin multiplicity is
      a per-system quantity.

      ``"system"`` must range from 0 to the number of systems given as input
      to the model.

  * - components
    -
    - the spin multiplicity must not have any components

  * - properties
    - ``"spin_multiplicity"``
    - the spin multiplicity must have a single property dimension named
      ``"spin_multiplicity"``, with a single entry set to ``0``.

The values represent the spin multiplicity :math:`2S + 1` of the system, where
:math:`S` is the total spin quantum number. The values are dimensionless and
stored as floats (matching the model's dtype), even though they always take
positive integer values. The value must be at least ``1``.

Common examples:

- ``1`` for a singlet (:math:`S = 0`)
- ``2`` for a doublet (:math:`S = 1/2`, e.g. a radical with one unpaired electron)
- ``3`` for a triplet (:math:`S = 1`)
