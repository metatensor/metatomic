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
from ._base import BaseModel


def execute_model(
    model: BaseModel,
    systems: Sequence[System],
    requested_outputs: Sequence[Quantity],
    *,
    selected_atoms: Optional[Labels] = None,
    check_consistency: bool = False,
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
    :return: the computed outputs, one :py:class:`metatensor.TensorMap` per requested
        output, in the same order
    """
    if not isinstance(model, BaseModel):
        raise TypeError(f"`model` must be a BaseModel, not {type(model)}")

    # non-owning view of the model, `model` is kept alive by the caller
    raw_model = model._as_mta_model_t()

    for system in systems:
        if not isinstance(system, System):
            raise TypeError(f"`systems` must be a list of System, not {type(system)}")

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

    outputs = (ctypes.POINTER(mts_tensormap_t) * len(requested_outputs))()

    lib = _get_library()
    lib.mta_execute_model(
        raw_model,
        systems_ptrs,
        len(systems_ptrs),
        selected_atoms_ptr,
        requested_outputs_json,
        check_consistency,
        outputs,
        len(outputs),
    )

    return [TensorMap.unsafe_from_ptr(output) for output in outputs]
