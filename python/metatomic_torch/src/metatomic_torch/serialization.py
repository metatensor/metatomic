import os
import pathlib
import warnings
from typing import BinaryIO, Union

import torch


if os.environ.get("METATOMIC_IMPORT_FOR_SPHINX", "0") != "0":
    from .documentation import System

else:
    System = torch.classes.metatomic.System

    _save = torch.ops.metatomic.save
    _load_system = torch.ops.metatomic.load_system

    _save_buffer = torch.ops.metatomic.save_buffer
    _load_system_buffer = torch.ops.metatomic.load_system_buffer


def load_system(file: str) -> System:
    """Load a System object from a file.

    The loaded System object will be on the CPU device and contain float64 data.

    :param file: The path (or file-like object) to load the System object from.
    :return: The System object.
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return _load_system(file)
    else:
        if isinstance(file, str):
            return _load_system(file)
        elif isinstance(file, pathlib.Path):
            return _load_system(str(file.resolve()))
        else:
            # assume a file-like object
            buffer = file.read()
            assert isinstance(buffer, bytes)

            with warnings.catch_warnings():
                # ignore warning about buffer beeing non-writeable
                warnings.simplefilter("ignore")
                buffer = torch.frombuffer(buffer, dtype=torch.uint8)

            return _load_system_buffer(buffer)


# modify the annotations in a way such that the TorchScript compiler does not see these,
# but sphinx does for documentation.
load_system.__annotations__["file"] = Union[str, pathlib.Path, BinaryIO]


def save(file: str, system: System) -> None:
    """Save a System object to a file.

    The provided System must contain float64 data and be on the CPU device.

    The saved file will be a zip archive containing the following files:

    - ``types.npy``, containing the atomic types in numpy's NPY format;
    - ``positions.npy``, containing the systems' positions in numpy's NPY format;
    - ``cell.npy``, containing the systems' cell in numpy's NPY format;
    - ``pbc.npy``, containing the periodic boundary conditions in numpy's NPY format;

    For each neighbor list in the System object, the following files will be saved
    (where ``{nl_idx}`` is the index of the neighbor list):

    - ``pairs/{nl_idx}/options.json``: the ``NeighborListOptions`` object
      converted to a JSON string.
    - ``pairs/{nl_idx}/data.mts``: the neighbor list ``TensorBlock`` object

    For each extra data in the System object, the following file will be saved (where
    ``{name}`` is the name of the extra data):

    - ``data/{name}.mts``: The extra data ``TensorMap``

    :param file: The path (or file-like object) to save the System to.
    :param system: The System object to save.
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return _save(file, system)
    else:
        if isinstance(file, str):
            return _save(file, system)
        elif isinstance(file, pathlib.Path):
            return _save(str(file.resolve()), system)
        else:
            # assume a file-like object
            buffer = _save_buffer(system)
            assert isinstance(buffer, torch.Tensor)
            file.write(buffer.numpy().tobytes())


# modify the annotations in a way such that the TorchScript compiler does not see these,
# but sphinx does for documentation.
save.__annotations__["file"] = Union[str, pathlib.Path, BinaryIO]


def load_system_buffer(buffer: torch.Tensor) -> System:
    """
    Load a previously saved :py:class:`System` from an in-memory buffer, stored
    inside a 1-dimensional :py:class:`torch.Tensor` of ``uint8``.

    :param buffer: CPU tensor with ``uint8`` dtype representing an in-memory buffer.
    """
    return _load_system_buffer(buffer)


def save_buffer(system: System) -> torch.Tensor:
    """
    Save the given ``system`` to an in-memory buffer, represented as a 1-dimensional
    :py:class:`torch.Tensor` with ``uint8`` dtype.

    :param system: The :py:class:`System` object to serialize and save.
    """
    return _save_buffer(system)
