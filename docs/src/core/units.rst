.. _units:

Units
^^^^^

Models in metatensor can use arbitrary units for their inputs and outputs. The
unit conversion system allows models to specify the units they expect and
receive data in any compatible unit, with automatic conversion handled by
during model execution.

Unit parsing is handled by one of the following functions:

- :c:func:`mta_unit_conversion_factor` in C
- :cpp:func:`metatomic::unit_conversion_factor` in C++

These functions parses two unit expressions, checks that they have compatible
physical dimensions, and returns the multiplicative conversion factor. For
example, in C++:

.. code-block:: C++

    // How many eV are in one kJ/mol?
    double factor = metatomic::unit_conversion_factor("kJ/mol", "eV");
    // factor ≈ 0.01036

    // How many GPa are in one eV/A^3?
    factor = metatomic::unit_conversion_factor("eV/A^3", "GPa");
    // factor ≈ 160.22

If either (or both) unit strings are empty, the conversion returns ``1.0``
without checking dimensions. This makes it safe to pass optional/unknown units.

.. _known-base-units:

Base units
~~~~~~~~~~

Unit expressions are built from the following base units. Matching is
case-insensitive, and whitespace is ignored.

**Temperature**:
  ``Kelvin`` (``K``)

**Length**:
  ``angstrom`` (``A``), ``Bohr``, ``meter`` (``m``), ``centimeter`` (``cm``),
  ``millimeter`` (``mm``), ``micrometer`` (``um``, ``µm``), ``nanometer`` (``nm``)

**Energy**:
  ``eV``, ``meV``, ``Hartree``, ``kcal``, ``kJ``, ``Joule`` (``J``), ``Rydberg`` (``Ry``)

**Time**:
  ``second`` (``s``), ``millisecond`` (``ms``), ``microsecond`` (``us``, ``µs``),
  ``nanosecond`` (``ns``), ``picosecond`` (``ps``), ``femtosecond`` (``fs``)

**Mass**:
  ``Dalton`` (``u``), ``kilogram`` (``kg``), ``gram`` (``g``), ``electron_mass`` (``m_e``)

**Charge**:
  ``e``, ``Coulomb`` (``C``)

**Pressure**:
  ``Pascal`` (``Pa``), ``kiloPascal`` (``kPa``), ``MegaPascal`` (``MPa``),
  ``GigaPascal`` (``GPa``), ``bar``, ``atm``

**Electric Dipole Moment**:
  ``Debye`` (``D``)

**Dimensionless**:
  ``mol``

**Derived constants**:
  ``hbar``

Expression syntax
~~~~~~~~~~~~~~~~~

Base units can be combined using the following operators:

- Multiplication: ``*`` or whitespace (``kJ mol``, ``kJ*mol``)
- Division: ``/`` (``kJ/mol``)
- Exponentiation: ``^`` (``A^3``, ``m^2``)
- Parentheses: ``()`` for grouping (``(eV*u)^(1/2)``)

Fractional powers
  Exponents can be integers (``A^3``) or fractions enclosed in parentheses
  (``^(1/2)``, ``^(2/3)``). Fractional powers are supported only when the
  result has integer physical dimensions — for example ``(eV*u)^(1/2)``
  computes momentum with dimensions :math:`[L T^{-1} M]`.

Numeric literals
  Bare numbers can be used as dimensionless quantity expressions, e.g.
  ``"2"`` evaluates to the conversion factor ``2.0``. This is useful when a
  model needs to define a unit that is simply a scalar multiple of another.

Examples of valid compound expressions:

- ``kJ/mol`` --- energy per mole
- ``eV/Angstrom^3`` or ``eV/A^3`` --- pressure
- ``(eV*u)^(1/2)`` --- momentum (fractional powers)
- ``Hartree/Bohr`` --- force in atomic units
- ``nm/fs`` --- velocity
