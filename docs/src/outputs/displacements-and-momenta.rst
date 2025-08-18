.. _positions-output:

Positions
^^^^^^^^^

Positions are differences between atomic positions at two different times.
They can be used to predict the next configuration in molecular dynamics
(see, e.g., https://arxiv.org/pdf/2505.19350).

In metatomic models, they are associated with the ``"positions"``
key in the model outputs, and must adhere to the following metadata schema:

.. list-table:: Metadata for positions
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single
      entry set to ``0``. Positions are always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]``
    - the samples must be named ``["system", "atom"]``, since
      positions are always per-atom.

      ``"system"`` must range from 0 to the number of systems given as an input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    - ``"xyz"``
    - positions must have a single component dimension named
      ``"xyz"``, with three entries set to ``0``, ``1``, and ``2``.  The
      positions are always 3D vectors, and the order of the
      components is x, y, z.

  * - properties
    - ``"positions"``
    - positions must have a single property dimension named
      ``"positions"``, with a single entry set to ``0``.

At the moment, positions are not integrated into any simulation engines.

.. _momenta-output:

Momenta
^^^^^^^

The momentum of a particle is a vector defined as its mass times its velocity.
Predictions of momenta can be used, for example, to predict a future step in molecular
dynamics (see, e.g., https://arxiv.org/pdf/2505.19350).

In metatomic models, they are associated with the ``"momenta"``
key in the model outputs, and must adhere to the following metadata schema:

.. list-table:: Metadata for momenta
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single
      entry set to ``0``. Momenta are always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]``
    - the samples must be named ``["system", "atom"]``, since
      momenta are always per-atom.

      ``"system"`` must range from 0 to the number of systems given as an input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    - ``"xyz"``
    - momenta must have a single component dimension named
      ``"xyz"``, with three entries set to ``0``, ``1``, and ``2``.  The
      momenta are always 3D vectors, and the order of the
      components is x, y, z.

  * - properties
    - ``"momenta"``
    - momenta must have a single property dimension named
      ``"momenta"``, with a single entry set to ``0``.

At the moment, momenta are not integrated into any simulation engines.
