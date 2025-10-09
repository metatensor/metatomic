Miscellaneous
=============

.. autofunction:: metatomic.torch.pick_device

.. autofunction:: metatomic.torch.unit_conversion_factor

.. _known-quantities-units:

Known quantities and units
--------------------------

The following quantities and units can be used with metatomic models. Adding new
units and quantities is very easy, please contact us if you need something else!
In the mean time, you can create :py:class:`metatomic.torch.ModelOutput` with
quantities that are not in this table. A warning will be issued and no unit
conversion will be performed.

When working with one of the quantities in this table, the unit you use must be
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
|   **momentum** | sqrt(eV*u)                                                                                        |
+----------------+---------------------------------------------------------------------------------------------------+
