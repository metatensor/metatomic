.. _output-variants:

Output variants
^^^^^^^^^^^^^^^

Models can provide multiple **variants** of the same output, for example
different exchange–correlation functionals for the energy. Users of a model can
then select which variant of the output should be used in simulation engines and
workflows. Variants are also sometimes referred to as **heads**, especially in
the context of deep learning models.

Variants are identified by appending ``"/<variant>"`` to the base output name.
For example:

- ``energy`` (default)
- ``energy/pbe``
- ``energy/pbe0``
- ``energy/r2scan``

.. important::

  If a model defines one or more variants, it **must also define the default
  base output** (e.g. ``energy``). Both the default and its variants follow the
  same :ref:`output metadata <atomistic-models-outputs>` rules.

-------------------------------------------------------------------------------

The following simulation engines can use variants for all their supported outputs:

.. grid:: 1 3 3 3

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    |ase-logo|
