Miscellaneous
=============

.. autofunction:: metatomic.torch.pick_device

.. autofunction:: metatomic.torch.pick_output

.. autofunction:: metatomic.torch.unit_conversion_factor

The set of recognized base units is documented in the
:cpp:func:`C++ API reference <metatomic_torch::unit_conversion_factor>`.

The :py:func:`unit_conversion_factor` function accepts any valid unit expression
built from base units combined with operators. There is no need to specify a
physical quantity --- the parser automatically verifies dimensional compatibility
between the source and target units.

.. _known-base-units:

Supported base units
~~~~~~~~~~~~~~~~~~~~

Unit expressions are built from the following base units. Matching is
case-insensitive, and whitespace is ignored.

**Length**:
  ``angstrom`` (``A``), ``Bohr``, ``meter`` (``m``), ``centimeter`` (``cm``),
  ``millimeter`` (``mm``), ``micrometer`` (``um``, ``µm``), ``nanometer`` (``nm``)

**Energy**:
  ``eV``, ``meV``, ``Hartree``, ``kcal``, ``kJ``, ``Joule`` (``J``), ``Rydberg`` (``Ry``)

**Time**:
  ``second`` (``s``), ``millisecond`` (``ms``), ``microsecond`` (``us``, ``µs``),
  ``nanosecond`` (``ns``), ``picosecond`` (``ps``), ``femtosecond`` (``fs``)

**Mass**:
  ``u`` (``Dalton``), ``kilogram`` (``kg``), ``gram`` (``g``), ``electron_mass`` (``m_e``)

**Charge**:
  ``e``, ``Coulomb`` (``C``)

**Dimensionless**:
  ``mol``

**Derived constants**:
  ``hbar``

Expression syntax
~~~~~~~~~~~~~~~~~~~

Base units can be combined using the following operators:

- Multiplication: ``*`` or whitespace (``kJ mol``, ``kJ*mol``)
- Division: ``/`` (``kJ/mol``)
- Exponentiation: ``^`` (``A^3``, ``m^2``)
- Parentheses: ``()`` for grouping (``(eV*u)^(1/2)``)

Examples of valid compound expressions:

- ``kJ/mol`` --- energy per mole
- ``eV/Angstrom^3`` or ``eV/A^3`` --- pressure
- ``(eV*u)^(1/2)`` --- momentum (fractional powers)
- ``Hartree/Bohr`` --- force in atomic units
- ``nm/fs`` --- velocity

The parser automatically checks that both unit expressions have matching
physical dimensions before computing the conversion factor.
