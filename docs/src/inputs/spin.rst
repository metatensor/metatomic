.. _spin-input:

Spin
^^^^

The spin multiplicity of the system is associated with the ``"spin"`` name,
and must have the following metadata:

.. list-table:: Metadata for spin input
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The spin is always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system"]``
    - the samples must be named ``["system"]``, since the spin multiplicity is
      a per-system quantity.

      ``"system"`` must range from 0 to the number of systems given as input
      to the model.

  * - components
    -
    - the spin must not have any components

  * - properties
    - ``"spin"``
    - the spin must have a single property dimension named ``"spin"``, with a
      single entry set to ``0``.

The values are integers representing the spin multiplicity :math:`2S + 1` of
the system, where :math:`S` is the total spin quantum number (e.g. ``1`` for a
singlet, ``2`` for a doublet, ``3`` for a triplet).
