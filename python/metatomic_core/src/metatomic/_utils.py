import ctypes
import re
import struct


_HEX_NUMBER = re.compile(r"(0[xX])?[0-9a-fA-F]+")


def string_from_mta(printed) -> str:
    """Read a ``mta_string_t`` as Python ``str`` and free the C string."""
    from ._c_lib import _get_library

    lib = _get_library()
    try:
        return lib.mta_string_view(printed).decode("utf8")
    finally:
        lib.mta_string_free(printed)


def hex_from_f64(value: float) -> str:
    """
    Get the hexadecimal representation of the bit pattern of ``value``.

    Storing floating point values as their bit pattern makes the JSON
    round-trip exact, without relying on the precision of the decimal
    representation. This also allows to store infinite values, which are not
    representable in JSON.
    """
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    return hex(bits)


def f64_from_hex(value: str, context: str) -> float:
    """
    Inverse of :py:func:`_hex_from_f64`, reading a ``f64`` from the hexadecimal
    representation of its bit pattern.

    ``context`` is used to build the error message if ``value`` is not a valid
    hexadecimal string.
    """
    if isinstance(value, str) and _HEX_NUMBER.fullmatch(value) is not None:
        bits = int(value, 16)
    else:
        bits = 2**64

    if bits >= 2**64:
        raise ValueError(f"{context} must be a hex-encoded string, got '{value}'")

    return struct.unpack("<d", struct.pack("<Q", bits))[0]


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
