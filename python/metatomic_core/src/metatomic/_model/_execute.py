import ctypes
import json
from collections.abc import Sequence
from typing import Optional

from metatensor import Labels, TensorMap
from metatensor._c_api import mts_tensormap_t

from .._c_api import mta_system_t
from .._c_lib import _get_library
from .._quantity import Quantity
from .._system import System
from .._utils import default_stream
from ._base import BaseModel


def execute_model(
    model: BaseModel,
    systems: Sequence[System],
    requested_outputs: Sequence[Quantity],
    *,
    selected_atoms: Optional[Labels] = None,
    check_consistency: bool = False,
    stream: Optional[int] = None,
) -> list[TensorMap]:
    """
    Execute a model to compute the requested outputs for a set of systems.

    :param model: the model to execute.
    :param systems: systems to run the model on
    :param requested_outputs: outputs the model should compute
    :param selected_atoms: optional selection of atoms to compute outputs for, or
        ``None`` to use all atoms
    :param check_consistency: if ``True``, run additional checks on the inputs and on
        the data produced by the model
    :param stream: stream on which the data of the systems is ready, and on which the
        outputs will be ready, see :c:func:`mta_execute_model` for the possible values.
    :return: the computed outputs, one :py:class:`metatensor.TensorMap` per requested
        output, in the same order. The outputs use the same array backend as the
        systems.
    """
    if not isinstance(model, BaseModel):
        raise TypeError(f"`model` must be a BaseModel, not {type(model)}")

    # non-owning view of the model, `model` is kept alive by the caller
    raw_model = model._as_mta_model_t()

    for system in systems:
        if not isinstance(system, System):
            raise TypeError(f"`systems` must be a list of System, not {type(system)}")

    backends = list(set(system.arrays_backend for system in systems))
    if len(backends) > 1:
        raise ValueError(
            "all systems must use the same arrays backend, got "
            + ", ".join(sorted(str(backend) for backend in backends))
        )

    if backends == ["torch"]:
        output_array_backend = "torch"
    else:
        output_array_backend = "numpy"

    for output in requested_outputs:
        if not isinstance(output, Quantity):
            raise TypeError(
                f"`requested_outputs` must be a list of Quantity, not {type(output)}"
            )

    systems_ptrs = (ctypes.POINTER(mta_system_t) * len(systems))(
        *[system.as_mta_system_t() for system in systems]
    )

    if selected_atoms is None:
        selected_atoms_ptr = None
    elif isinstance(selected_atoms, Labels):
        selected_atoms_ptr = selected_atoms.as_mts_labels_t()
    else:
        raise TypeError(
            f"`selected_atoms` must be metatensor Labels or None, "
            f"not {type(selected_atoms)}"
        )

    requested_outputs_json = json.dumps(
        [output.to_dict() for output in requested_outputs]
    ).encode("utf8")

    if stream is None and len(systems) != 0 and systems[0].arrays_backend == "torch":
        stream = default_stream(systems[0].positions)

    outputs = (ctypes.POINTER(mts_tensormap_t) * len(requested_outputs))()

    lib = _get_library()
    lib.mta_execute_model(
        raw_model,
        systems_ptrs,
        len(systems_ptrs),
        selected_atoms_ptr,
        None if stream is None else ctypes.byref(ctypes.c_int64(stream)),
        requested_outputs_json,
        check_consistency,
        outputs,
        len(outputs),
    )

    return [
        TensorMap.unsafe_from_ptr(output, array_backend=output_array_backend)
        for output in outputs
    ]
