.. _momentum-quantity:

Momentum
^^^^^^^^

The momentum of a particle is a vector defined as its mass times its velocity.
Predictions of momenta can be used, for example, to predict a future step in
molecular dynamics (see, e.g., https://arxiv.org/pdf/2505.19350).

In metatomic models, they are associated with the ``"momentum"`` or
``"momentum/<variant>"`` name (see :ref:`quantity-variants`), and must have the
following metadata:

.. list-table:: Metadata for ``"momentum"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The ``"momentum"`` quantity is always represented as a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]``
    - the samples must be named ``["system", "atom"]``, since ``"momentum"`` is
      always per-atom.

      ``"system"`` must range from 0 to the number of systems given as input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    - ``"xyz"``
    - The ``"momentum"`` quantity must have a single component dimension named
      ``"xyz"``, with three entries set to ``0``, ``1``, and ``2``. The momentum
      is always a 3D vector, and the order of the components is ``x, y, z``.

  * - properties
    - ``"momentum"``
    - The ``"momentum"`` quantity must have a single property dimension named
      ``"momentum"``, with a single entry set to ``0``.

The following simulation engine can provide ``"momentum"`` as inputs to the
models:

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
