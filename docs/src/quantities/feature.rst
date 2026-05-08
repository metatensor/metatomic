.. _feature-quantity:

Features
^^^^^^^^

Features are numerical vectors representing a given structure or
atom/atom-centered environment in an abstract n-dimensional space. They are also
sometimes called descriptors, representations, embeddings, *etc.*

Features can be computed with an analytical expression (for example `SOAP
power spectrum`_, `atom-centered symmetry functions`_, …), or learned indirectly
by a neural-network or a similar machine learning construct.

.. _SOAP power spectrum: https://doi.org/10.1103/PhysRevB.87.184115
.. _Atom-centered symmetry functions: https://doi.org/10.1063/1.3553717

In metatomic models, they are associated with the ``"feature"`` or
``"feature/<variant>"`` name (see :ref:`quantity-variants`), and must have the
following metadata:

.. list-table:: Metadata for ``"feature"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The ``"feature"`` quantity is always represented as a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]`` or ``["system"]``
    - the samples should be named ``["system", "atom"]`` for per-atom quantities;
      or ``["system"]`` for per-system quantities.

      ``"system"`` must range from 0 to the number of systems given as input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    -
    - the ``"feature"`` quantity must not have any components.

  * - properties
    -
    - the ``"feature"`` quantity can have arbitrary properties.

.. note::
  Features are typically handled without a unit, so the ``"unit"`` field of
  :py:func:`metatomic.torch.ModelOutput` is typically left empty.

The following simulation engines can use the ``"feature"`` quantity as an
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
    :link: engine-chemiscope
    :link-type: ref

    |chemiscope-logo|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-plumed
    :link-type: ref

    |plumed-logo|


Gradients of the ``"feature"`` quantity
---------------------------------------

The ``"feature"`` quantity is typically used with automatic differentiation for
the gradients, and explicit gradients are not currently specified.
