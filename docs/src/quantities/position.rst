.. _position-quantity:

Position
^^^^^^^^

Atomic positions are associated with the ``"position"`` or
``"position/<variant>"`` name (see :ref:`quantity-variants`), and must have the
following metadata:

.. list-table:: Metadata for ``"position"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The ``"position"`` quantity is always represented as a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]``
    - the samples must be named ``["system", "atom"]``, since ``"position"`` is
      always per-atom.

      ``"system"`` must range from 0 to the number of systems given as input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    - ``"xyz"``
    - The ``"position"`` quantity must have a single component dimension named
      ``"xyz"``, with three entries set to ``0``, ``1``, and ``2``. The position
      is always a 3D vector, and the order of the components is ``x, y, z``.

  * - properties
    - ``"position"``
    - The ``"position"`` quantity must have a single property dimension named
      ``"position"``, with a single entry set to ``0``.

The following simulation engines can use the ``"position"`` quantity as output:

.. grid:: 1 3 3 3

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-lammps
    :link-type: ref

    |lammps-logo|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ipi
    :link-type: ref

    |ipi-logo|
