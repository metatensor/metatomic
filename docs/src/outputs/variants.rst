.. _output-variants:

Output variants
^^^^^^^^^^^^^^^

Models can provide multiple *variants* of the same output, for example different
exchange–correlation functionals for the energy. Variants allow you to select which
"head" of the model to use in downstream engines and workflows.

Variants are identified by appending ``"/<variant>"`` to the base output name. For
example:

- ``energy`` (default)
- ``energy/pbe0``
- ``energy/r2scan``

If a model defines one or more variants, it **must also define the default base output**
(e.g. ``energy``). Both the default and its variants follow the same :ref:`output
metadata <atomistic-models-outputs>` rules.

The following simulation engines currently support output variants:

.. grid:: 1 3 3 3

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    |ase-logo|
