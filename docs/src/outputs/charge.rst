.. _charge-output:

Charge
^^^^^^

The total charge of the system is associated with the ``"charge"`` name, and
must have the following metadata:

.. list-table:: Metadata for charge output
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The charge is always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system"]``
    - the samples must be named ``["system"]``, since the charge is a
      per-system quantity. When running a batched calculation, there will be
      one row per system.

      ``"system"`` must range from 0 to the number of systems given as input
      to the model.

  * - components
    -
    - the charge must not have any components

  * - properties
    - ``"charge"``
    - the charge must have a single property dimension named ``"charge"``,
      with a single entry set to ``0``.

The values represent the total electric charge of the system in units of the
elementary charge :math:`e` (e.g. ``0`` for a neutral system, ``-1`` for a
singly charged anion). The values are stored as floats (matching the model's
dtype), even though they typically take integer values. The unit is always
``"e"`` (elementary charges).

The following simulation engines support the ``"charge"`` output:

.. grid:: 1 1 1 1

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    |ase-logo|

In ASE, the charge is read from ``atoms.info["charge"]`` and defaults to
``0`` (neutral) if not set.
