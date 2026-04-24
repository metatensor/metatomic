.. _charge-quantity:

Charge
^^^^^^

Electric charges are associated with the ``"charge"`` or ``"charge/<variant>"``
name (see :ref:`quantity-variants`), and must have the following metadata:

.. list-table:: Metadata for ``"charge"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The ``"charge"`` quantity is always represented as a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]`` or ``["system"]``
    - the samples should be named ``["system", "atom"]`` for per-atom charges
      (one value per atom), or ``["system"]`` for the per-system total charge
      (a single value per system). The form is selected by the requested
      :py:attr:`~metatomic.torch.ModelOutput.sample_kind` (``"atom"`` or
      ``"system"``).

      ``"system"`` must range from 0 to the number of systems given as input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    -
    - the ``"charge"`` quantity must not have any components

  * - properties
    - ``"charge"``
    - the ``"charge"`` quantity must have a single property dimension named
      ``"charge"``, with a single entry set to ``0``.


The following simulation engine can provide ``"charge"`` as inputs to the
models:

.. grid:: 1 3 3 3

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    |ase-logo|
