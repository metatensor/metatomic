.. _non-conservative-force-quantity:

Non-conservative force
^^^^^^^^^^^^^^^^^^^^^^

Non-conservative forces are forces that are not calculated as the negative
gradient of a potential energy function, but rather directly predicted by the
model, without going through the potential energy. These forces are generally
faster to compute than forces derived from the potential energy by automatic
differentiation. However, these predictions must be used with care, see
https://arxiv.org/abs/2412.11569.

In metatomic models, they are associated with the ``"non_conservative_force"``
or ``"non_conservative_force/<variant>"`` name (see :ref:`quantity-variants`),
and must have the following metadata:

.. list-table:: Metadata for ``"non_conservative_force"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The ``"non_conservative_force"`` quantity is always
      represented as a :py:class:`metatensor.torch.TensorMap` with a single
      block.

  * - samples
    - ``["system", "atom"]``
    - the samples must be named ``["system", "atom"]``, since
      ``"non_conservative_force"`` is always per-atom.

      ``"system"`` must range from 0 to the number of systems given as input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    - ``"xyz"``
    - The ``"non_conservative_force"`` quantity must have a single component
      dimension named ``"xyz"``, with three entries set to ``0``, ``1``, and
      ``2``.  The non-conservative force is always a 3D vector, and the order
      of the components is ``x, y, z``.

  * - properties
    - ``"non_conservative_force"``
    - The ``"non_conservative_force"`` quantity must have a single property
      dimension named ``"non_conservative_force"``, with a single entry set to
      ``0``.

The following simulation engines can use the ``"non_conservative_force"``
output, typically using a ``non_conservative`` flag:

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

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-lammps
    :link-type: ref

    |lammps-logo|

.. note::

    If you are adding support for ``non_conservative_force`` in a molecular
    dynamics engine, metatomic models might predict a non zero total force. You
    should consider removing this total force to prevent drift in your
    simulations.

.. _non-conservative-stress-quantity:

Non-conservative stress
^^^^^^^^^^^^^^^^^^^^^^^

Similar to the ``"non_conservative_force"``, the non-conservative stress is a
stress tensor that is not calculated using derivatives of the potential energy,
but directly predicted by the model. As with forces, they are typically faster
to compute but need to be used with care, see https://arxiv.org/abs/2412.11569.

In metatomic models, they are associated with the ``"non_conservative_stress"``
or ``"non_conservative_stress/<variant>"`` name (see :ref:`quantity-variants`),
and must have the following metadata:

.. list-table:: Metadata for ``"non_conservative_stress"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The ``"non_conservative_force"`` quantity is always
      represented as a :py:class:`metatensor.torch.TensorMap` with a single
      block.

  * - samples
    - ``"system"``
    - the samples must be named ``["system"]``, since
      ``"non_conservative_stress"`` is always per-system.

      ``"system"`` must range from 0 to the number of systems given as input
      to the model.

  * - components
    - ``["xyz_1"], ["xyz_2"]``
    - the ``"non_conservative_stress"`` quantity must have two components labels
      with ``"xyz_1"`` and ``"xyz_2"`` as their respective names, both with
      three entries set to ``0``, ``1``, and ``2``. The order of the components
      along both directions is ``x, y, z``.

  * - properties
    - ``"non_conservative_stress"``
    - the ``"non_conservative_stress"`` quantity must have a single property
      dimension named ``"non_conservative_stress"``, with a single entry set to
      ``0``.

The following simulation engines can use the ``"non_conservative_stress"``
output, typically using a ``non_conservative`` flag:

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

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-lammps
    :link-type: ref

    |lammps-logo|
