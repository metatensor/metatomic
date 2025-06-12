Models
======

Most of the code in ``metatomic.torch`` is here to define and export
models and to store model metadata. The corresponding classes are documented
below:

.. toctree::
    :maxdepth: 1

    export
    metadata

We also provide a couple of functions to work with the models:

.. autofunction:: metatomic.torch.read_model_metadata

.. autofunction:: metatomic.torch.load_atomistic_model

.. autofunction:: metatomic.torch.check_atomistic_model

.. autofunction:: metatomic.torch.load_model_extensions

.. autofunction:: metatomic.torch.unit_conversion_factor

.. _known-quantities-units:

Known quantities and units
--------------------------

The following quantities and units can be used with metatomic models. Adding new
units and quantities is very easy, please contact us if you need something else!
In the mean time, you can create :py:class:`metatomic.torch.ModelOutput` with
quantities that are not in this table. A warning will be issued and no unit
conversion will be performed.

When working with one of the quantity in this table, the unit you use must be
one of the registered unit.

+----------------+---------------------------------------------------------------------------------------------------+
|   quantity     | units                                                                                             |
+================+===================================================================================================+
|   **length**   | angstrom (A), Bohr, meter, centimeter (cm), millimeter (mm), micrometer (um, µm), nanometer (nm)  |
+----------------+---------------------------------------------------------------------------------------------------+
|   **energy**   | eV, meV, Hartree, kcal/mol, kJ/mol, Joule (J), Rydberg (Ry)                                       |
+----------------+---------------------------------------------------------------------------------------------------+
|   **force**    | eV/Angstrom (eV/A, eV/Angstrom)                                                                   |
+----------------+---------------------------------------------------------------------------------------------------+
|   **pressure** | eV/Angstrom^3 (eV/A^3, eV/Angstrom^3)                                                             |
+----------------+---------------------------------------------------------------------------------------------------+
