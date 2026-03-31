.. _mass-quantity:

Mass
^^^^

The mass of atomistic objects are associated with the ``"mass"`` or
``"mass/<variant>"`` name (see :ref:`quantity-variants`), and must have the
following metadata:

.. list-table:: Metadata for ``"mass"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The ``"mass"`` quantity is always represented as a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]``
    - the samples must be named ``["system", "atom"]``, since ``"mass"`` is
      always per-atom.

      ``"system"`` must range from 0 to the number of systems given as input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    -
    - the ``"mass"`` quantity must not have any components

  * - properties
    - ``"mass``
    - The ``"mass"`` quantity must have a single property dimension named
      ``"mass"``, with a single entry set to ``0``.


The following simulation engine can provide ``"mass"`` as inputs to the models:

.. grid:: 1 3 3 3

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    |ase-logo|
