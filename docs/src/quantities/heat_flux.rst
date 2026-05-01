.. _heat-flux-quantity:

Heat Flux
^^^^^^^^^

Heat flux is associated with the ``"heat_flux"`` or ``"heat_flux/<variant>"``
name (see :ref:`quantity-variants`), and must have the following metadata:

.. list-table:: Metadata for ``"heat_flux"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The ``"heat_flux"`` quantity is always represented as a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system"]``
    - the samples must be named ``["system"]``, since ``"heat_flux"`` is
      always per-system.

      ``"system"`` must range from 0 to the number of systems given as input
      to the model.

  * - components
    - ``"xyz"``
    - The ``"heat_flux"`` quantity must have a single component dimension named
      ``"xyz"``, with three entries set to ``0``, ``1``, and ``2``. The heat
      flux is always a 3D vector, and the order of the components is ``x, y,
      z``.

  * - properties
    - ``"heat_flux"``
    - The ``"heat_flux"`` quantity must have a single property dimension named
      ``"heat_flux"``, with a single entry set to ``0``.


The following simulation engine can use the ``"heat_flux"`` quantity as an
output:

.. grid:: 1 3 3 3

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    |ase-logo|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ipi
    :link-type: ref

    |ipi-logo|
