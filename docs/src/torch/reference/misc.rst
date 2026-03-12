Miscellaneous
=============

.. autofunction:: metatomic.torch.pick_device

.. autofunction:: metatomic.torch.pick_output

.. autofunction:: metatomic.torch.unit_conversion_factor

The set of recognized base units is documented in the
:cpp:func:`C++ API reference <metatomic_torch::unit_conversion_factor>`.

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

.. list-table:: Supported Units by Quantity
   :header-rows: 1
   :widths: 15 85

   * - Quantity
     - Units
   * - **length**
     - ``angstrom`` (``A``), ``Bohr``, ``meter`` (``m``), ``centimeter`` (``cm``),
       ``millimeter`` (``mm``), ``micrometer`` (``um``, ``µm``), ``nanometer`` (``nm``)
   * - **energy**
     - ``eV``, ``meV``, ``Hartree``, ``kcal/mol``, ``kJ/mol``, ``Joule`` (``J``),
       ``Rydberg`` (``Ry``)
   * - **force**
     - ``eV/Angstrom`` (``eV/A``), ``Hartree/Bohr``
   * - **pressure**
     - ``eV/Angstrom^3`` (``eV/A^3``)
   * - **momentum**
     - ``u*A/fs``, ``u*A/ps``, ``(eV*u)^(1/2)``, ``kg*m/s``, ``hbar/Bohr``
   * - **mass**
     - ``u`` (``Dalton``), ``kg`` (``kilogram``), ``g`` (``gram``),
       ``electron_mass`` (``m_e``)
   * - **velocity**
     - ``nm/fs``, ``A/fs``, ``m/s``, ``nm/ps``, ``Bohr*Hartree/hbar``
   * - **charge**
     - ``e``, ``Coulomb`` (``C``)
   * - **time**
     - ``second`` (``s``), ``millisecond`` (``ms``), ``microsecond`` (``us``, ``µs``),
       ``nanosecond`` (``ns``), ``picosecond`` (``ps``), ``femtosecond`` (``fs``)
