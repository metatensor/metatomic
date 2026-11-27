import ctypes
import re
import struct
import sys
from typing import Optional

from ctypes_dlpack import DLDevice


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


def synchronize_torch_mps(array):
    """
    Wait for all the work submitted by torch on MPS devices to finish, if ``array``
    is a torch tensor on an MPS device.

    DLPack does not define how to synchronize data on Metal devices, and torch does
    not expose the command queue it uses for MPS devices, so this is the only way to
    make sure the data in ``array`` is ready before using it outside of torch.
    """
    # if torch was never imported, `array` can not be a torch tensor
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(array, torch.Tensor):
        return

    if array.device.type == "mps":
        torch.mps.synchronize()


def default_stream(array) -> Optional[int]:
    """
    Get the stream to use with the data in ``array`` when no stream was given by the
    user, following the conventions of :c:func:`mta_execute_model`.

    For torch tensors on CUDA devices, this is the current torch stream. For torch
    tensors on MPS devices, this waits for all the work submitted by torch to finish
    (see :py:func:`synchronize_torch_mps`), and returns ``None``. This function
    returns ``None`` for all other arrays.
    """
    synchronize_torch_mps(array)

    torch = sys.modules.get("torch")
    if torch is None or not isinstance(array, torch.Tensor):
        return None

    if array.device.type == "cuda":
        stream = torch.cuda.current_stream(array.device).cuda_stream
        # torch uses 0 for the legacy default stream, while DLPack uses 1
        return 1 if stream == 0 else stream

    return None


def stream_synchronize(device, stream: Optional[int]):
    """
    Block the calling thread until all the work submitted to ``stream`` on
    ``device`` is complete.

    This does nothing if ``stream`` is ``None``. See :c:func:`mta_execute_model` for
    the possible values of ``stream``.

    :param device: the device on which the stream lives, either as a
        :py:class:`ctypes_dlpack.DLDevice` or as a ``(device_type, device_id)`` tuple
        (as returned by ``array.__dlpack_device__()``)
    :param stream: the stream to synchronize, or ``None``
    """
    from ._c_lib import _get_library

    if not isinstance(device, DLDevice):
        device_type, device_id = device
        device = DLDevice(device_type=int(device_type), device_id=device_id)

    _get_library().mta_stream_synchronize(
        device,
        None if stream is None else ctypes.byref(ctypes.c_int64(stream)),
    )
