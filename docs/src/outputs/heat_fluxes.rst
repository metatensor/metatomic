.. _heat-fluxes-output:

Heat Fluxes
^^^^^^^^^^^

Heat fluxes are associated with the ``"heat_flux"`` or
``"heat_flux/<variant>"`` name (see :ref:`output-variants`), and must have the
following metadata:

.. list-table:: Metadata for heat fluxes
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single
      entry set to ``0``. Heat fluxes are always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system"]``
    - the samples must be named ``["system"]``, since
      heat fluxes are always not per-atom.

      ``"system"`` must range from 0 to the number of systems given as an input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    - ``"xyz"``
    - heat fluxes must have a single component dimension named
      ``"xyz"``, with three entries set to ``0``, ``1``, and ``2``.  The
      heat fluxes are always 3D vectors, and the order of the
      components is x, y, z.

  * - properties
    - ``"heat_flux"``
    - heat fluxes must have a single property dimension named
      ``"heat_flux"``, with a single entry set to ``0``.

The following simulation engine can use the ``"heat_flux"`` output.

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
