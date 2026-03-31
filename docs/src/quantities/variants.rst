.. _quantity-variants:

Variants
^^^^^^^^

Models can provide multiple **variants** of the same quantity, for example
different exchange–correlation functionals for the energy. Users of a model can
then select which variant of the quantity should be used in simulation engines
and workflows. Variants are also sometimes referred to as **heads**, especially
in the context of deep learning models outputs.

Variants are identified by appending ``"/<variant>"`` to the base output name.
For example:

- ``energy`` (default)
- ``energy/pbe``
- ``energy/pbe0``
- ``energy/r2scan``
