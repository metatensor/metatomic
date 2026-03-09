Miscellaneous
=============

.. autofunction:: metatomic.torch.pick_device

.. autofunction:: metatomic.torch.pick_output

.. autofunction:: metatomic.torch.unit_conversion_factor

.. _known-quantities-units:

Unit expression parser
----------------------

``unit_conversion_factor`` accepts arbitrary unit expressions built from base
tokens combined with ``*``, ``/``, ``^``, and parentheses. For example:

- ``"kJ/mol"``
- ``"eV/Angstrom^3"``
- ``"(eV*u)^(1/2)"``
- ``"Hartree/Bohr"``

Dimensional compatibility is verified automatically; no ``quantity`` parameter
is needed. Token lookup is case-insensitive, and whitespace is ignored.

Base unit tokens
^^^^^^^^^^^^^^^^

.. list-table:: Supported Unit Tokens
   :header-rows: 1

   * - Dimension
     - Tokens
     - Notes
   * - **Length**
     - ``angstrom`` (``a``), ``bohr``, ``nm`` (``nanometer``), ``meter`` (``m``), ``cm`` (``centimeter``), ``mm`` (``millimeter``), ``um`` (``micrometer``)
     -
   * - **Energy**
     - ``ev``, ``mev``, ``hartree``, ``ry`` (``rydberg``), ``joule`` (``j``), ``kcal``, ``kj``
     - ``kcal`` and ``kj`` are bare (not per-mol); write ``kcal/mol`` for the per-mole unit
   * - **Time**
     - ``s`` (``second``), ``ms`` (``millisecond``), ``us`` (``microsecond``), ``ns`` (``nanosecond``), ``ps`` (``picosecond``), ``fs`` (``femtosecond``)
     -
   * - **Mass**
     - ``u`` (``dalton``), ``kg`` (``kilogram``), ``g`` (``gram``), ``electron_mass`` (``m_e``)
     -
   * - **Charge**
     - ``e``, ``coulomb`` (``c``)
     -
   * - **Dimensionless**
     - ``mol``
     - Avogadro scaling factor
   * - **Derived**
     - ``hbar``
     - :math:`\hbar` in SI (:math:`M L^2 T^{-1}`)

Known quantities
^^^^^^^^^^^^^^^^

When setting ``quantity`` on a :py:class:`~metatomic.torch.ModelOutput`, the
following names are recognized. The parser will check that the unit expression
has dimensions matching the expected quantity.

.. list-table:: Physical Dimensions
   :header-rows: 1

   * - quantity
     - expected dimension
   * - **length**
     - :math:`L`
   * - **energy**
     - :math:`M L^2 T^{-2}`
   * - **force**
     - :math:`M L T^{-2}`
   * - **pressure**
     - :math:`M L^{-1} T^{-2}`
   * - **momentum**
     - :math:`M L T^{-1}`
   * - **mass**
     - :math:`M`
   * - **velocity**
     - :math:`L T^{-1}`
   * - **charge**
     - :math:`Q`

.. note::

   The 3-argument form ``unit_conversion_factor(quantity, from_unit, to_unit)``
   is deprecated. Use the 2-argument form instead. The ``quantity`` parameter is
   ignored by the parser; dimensional compatibility is checked automatically.
