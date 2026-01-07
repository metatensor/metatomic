.. _velocities-output:

Velocities
^^^^^^^^^^

Velocities are associated with the ``"velocities"`` or
``"velocities/<variant>"`` key (see :ref:`output-variants`) key in the model
outputs, and must adhere to the following metadata schema:

.. list-table:: Metadata for velocities
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
      velocities are always per-atom.

      ``"system"`` must range from 0 to the number of systems given as an input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    - ``"xyz"``
    - velocities must have a single component dimension named
      ``"xyz"``, with three entries set to ``0``, ``1``, and ``2``.  The
      velocities are always 3D vectors, and the order of the
      components is x, y, z.

  * - properties
    - ``"velocities"``
    - velocities must have a single property dimension named
      ``"velocities"``, with a single entry set to ``0``.

At the moment, velocities are not integrated into any simulation engines.
