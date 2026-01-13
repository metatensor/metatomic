.. _masses-output:

Masses
^^^^^^

Masses are associated with the ``"masses"`` key in the model
inputs, and must adhere to the following metadata schema:

.. list-table:: Metadata for masses
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single
      entry set to ``0``. Masses are always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]``
    - the samples must be named ``["system", "atom"]``, since
      masses are always per-atom.

      ``"system"`` must range from 0 to the number of systems given as an input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    -
    - the masses must not have any components

  * - properties
    - ``"masses"``
    - masses must have a single property dimension named
      ``"masses"``, with a single entry set to ``0``.

The following simulation engine can use the ``"masses"`` as inputs to the models.

.. grid:: 1 3 3 3

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    |ase-logo|

