import ctypes


def unit_conversion_factor(from_unit: str, to_unit: str) -> float:
    """Return the conversion factor between compatible units.

    Unit expressions follow :ref:`units`. If either unit is empty, return 1.0.
    Incompatible or unknown units raise :py:class:`metatomic.MetatomicError`.
    """
    from ._c_lib import _get_library

    factor = ctypes.c_double()
    _get_library().mta_unit_conversion_factor(
        from_unit.encode("utf8"), to_unit.encode("utf8"), ctypes.byref(factor)
    )
    return factor.value


def _string_from_mta(printed) -> str:
    """Read a ``mta_string_t`` as Python ``str`` and free the C string."""
    from ._c_lib import _get_library

    lib = _get_library()
    try:
        return lib.mta_string_view(printed).decode("utf8")
    finally:
        lib.mta_string_free(printed)
