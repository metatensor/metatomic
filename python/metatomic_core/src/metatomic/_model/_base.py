import abc
import ctypes
import json
from typing import Optional

from metatensor import Labels, TensorMap
from metatensor._c_lib import _get_library as _get_mts_library

from .._c_api import mta_model_t, mta_status_t
from .._c_lib import _get_library
from .._capabilities import ModelCapabilities
from .._metadata import ModelMetadata
from .._quantity import Quantity
from .._status import save_exception
from .._system import PairListOptions, System


class BaseModel(abc.ABC):
    """
    Abstract base class for atomistic models.

    This class provides a Python interface for implementing custom models. Users can
    inherit from this class, override the abstract methods, and then run the model with
    :py:func:`execute_model`.
    """

    @abc.abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Get the capabilities of this model."""

    @abc.abstractmethod
    def metadata(self) -> ModelMetadata:
        """Get metadata describing this model."""

    @abc.abstractmethod
    def requested_pair_lists(self) -> list[PairListOptions]:
        """List the pair lists (neighbor lists) this model needs as input."""

    @abc.abstractmethod
    def requested_inputs(self) -> list[Quantity]:
        """List the additional per-system inputs this model needs."""

    @abc.abstractmethod
    def execute_inner(
        self,
        systems: list[System],
        selected_atoms: Optional[Labels],
        requested_outputs: list[Quantity],
    ) -> list[TensorMap]:
        """
        Run the model and compute the requested outputs.

        This method should not be used directly. It is intended to be used through
        :py:func:`execute_model`, which handles unit conversion and can check inputs and
        outputs for consistency.

        The ``systems`` are non-owning views of systems owned by the caller, and their
        :py:attr:`System.arrays_backend` is not set. Implementations should set it
        before accessing the positions, cell, etc.

        :param systems: systems to run the model on
        :param selected_atoms: optional selection of atoms to compute outputs for, or
            ``None`` to use all atoms
        :param requested_outputs: outputs the model should compute
        :return: the computed outputs, one :py:class:`metatensor.TensorMap` per
            requested output, in the same order
        """

    def _as_mta_model_t(self) -> mta_model_t:
        """
        Build a ``mta_model_t`` pointing at this model, without taking ownership of it.
        This is used by :py:func:`execute_model` to run the model through
        ``mta_execute_model``.

        The ``unload`` callback of the returned ``mta_model_t`` is left as ``NULL``, and
        none of the other callbacks free the model.

        .. warning::

            The returned ``mta_model_t`` is a view of this model: it stores a plain
            pointer to it and does nothing to keep it alive. It is the caller's
            responsibility to ensure this model outlives every use of the returned
            ``mta_model_t``, and to never pass the result to an API that takes ownership
            of the model (i.e. one that would call ``unload``).
        """
        model = mta_model_t()
        model.data = id(self)
        model.unload = _CALLBACK_TYPES["unload"]()
        model.capabilities = _MTA_MODEL_PY_CAPABILITIES
        model.metadata = _MTA_MODEL_PY_METADATA
        model.requested_pair_lists = _MTA_MODEL_PY_REQUESTED_PAIR_LISTS
        model.requested_inputs = _MTA_MODEL_PY_REQUESTED_INPUTS
        model.execute_inner = _MTA_MODEL_PY_EXECUTE_INNER
        return model


### ================================================================================ ###

# Implementation of the `mta_model_t` callbacks for Python models. The `data` pointer
# of the model contains a borrowed `PyObject*` to the corresponding `BaseModel`.


def _model_from_data(model_data) -> BaseModel:
    return ctypes.cast(model_data, ctypes.py_object).value


def _set_json_output(output, value):
    lib = _get_library()
    output[0] = lib.mta_string_create(json.dumps(value).encode("utf8"))


def _check_list_of(values, cls, function: str):
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{function}() must return a list, got {type(values)}")

    for value in values:
        if not isinstance(value, cls):
            raise TypeError(
                f"{function}() must return a list of {cls.__name__}, "
                f"got an element of type {type(value)}"
            )


def _mta_model_capabilities(model_data, capabilities_json):
    try:
        capabilities = _model_from_data(model_data).capabilities()
        if not isinstance(capabilities, ModelCapabilities):
            raise TypeError(
                "capabilities() must return a ModelCapabilities, "
                f"got {type(capabilities)}"
            )
        _set_json_output(capabilities_json, capabilities.to_dict())
        return mta_status_t.MTA_SUCCESS
    except BaseException as e:
        save_exception(e)
        return mta_status_t.MTA_MODEL_ERROR


def _mta_model_metadata(model_data, metadata_json):
    try:
        metadata = _model_from_data(model_data).metadata()
        if not isinstance(metadata, ModelMetadata):
            raise TypeError(
                f"metadata() must return a ModelMetadata, got {type(metadata)}"
            )
        _set_json_output(metadata_json, metadata.to_dict())
        return mta_status_t.MTA_SUCCESS
    except BaseException as e:
        save_exception(e)
        return mta_status_t.MTA_MODEL_ERROR


def _mta_model_requested_pair_lists(model_data, pair_options_json):
    try:
        pair_lists = _model_from_data(model_data).requested_pair_lists()
        _check_list_of(pair_lists, PairListOptions, "requested_pair_lists")
        _set_json_output(pair_options_json, [p.to_dict() for p in pair_lists])
        return mta_status_t.MTA_SUCCESS
    except BaseException as e:
        save_exception(e)
        return mta_status_t.MTA_MODEL_ERROR


def _mta_model_requested_inputs(model_data, inputs_json):
    try:
        inputs = _model_from_data(model_data).requested_inputs()
        _check_list_of(inputs, Quantity, "requested_inputs")
        _set_json_output(inputs_json, [i.to_dict() for i in inputs])
        return mta_status_t.MTA_SUCCESS
    except BaseException as e:
        save_exception(e)
        return mta_status_t.MTA_MODEL_ERROR


def _mta_model_execute_inner(
    model_data,
    systems,
    systems_count,
    selected_atoms,
    requested_outputs_json,
    outputs,
    outputs_count,
):
    system_views = []
    try:
        model = _model_from_data(model_data)

        system_views = [
            System.unsafe_view_from_ptr(systems[i]) for i in range(systems_count)
        ]
        py_systems = system_views

        py_selected_atoms = None
        if selected_atoms:
            # `selected_atoms` is borrowed, so we give a copy to the model
            mts_lib = _get_mts_library()
            py_selected_atoms = Labels.unsafe_from_ptr(
                mts_lib.mts_labels_clone(selected_atoms)
            )

        requested_outputs = [
            Quantity.from_dict(item)
            for item in json.loads(requested_outputs_json.decode("utf8"))
        ]

        py_outputs = model.execute_inner(
            py_systems, py_selected_atoms, requested_outputs
        )
        _check_list_of(py_outputs, TensorMap, "execute_inner")

        if len(py_outputs) != outputs_count:
            raise ValueError(
                f"model returned {len(py_outputs)} outputs, "
                f"but {outputs_count} were requested"
            )

        for i, output in enumerate(py_outputs):
            outputs[i] = output.release()

        return mta_status_t.MTA_SUCCESS
    except BaseException as e:
        save_exception(e)
        return mta_status_t.MTA_MODEL_ERROR
    finally:
        # The systems are only valid during this call, but the views can outlive it
        # (e.g. through the traceback of an exception, or if the model keeps a
        # reference to them). Mark them as released, so that any later use raises an
        # error instead of accessing freed memory.
        for view in system_views:
            view._ptr = None


# Keep the ctypes function pointers alive for the whole lifetime of the process
_CALLBACK_TYPES = dict(mta_model_t._fields_)
_MTA_MODEL_PY_CAPABILITIES = _CALLBACK_TYPES["capabilities"](_mta_model_capabilities)
_MTA_MODEL_PY_METADATA = _CALLBACK_TYPES["metadata"](_mta_model_metadata)
_MTA_MODEL_PY_REQUESTED_PAIR_LISTS = _CALLBACK_TYPES["requested_pair_lists"](
    _mta_model_requested_pair_lists
)
_MTA_MODEL_PY_REQUESTED_INPUTS = _CALLBACK_TYPES["requested_inputs"](
    _mta_model_requested_inputs
)
_MTA_MODEL_PY_EXECUTE_INNER = _CALLBACK_TYPES["execute_inner"](_mta_model_execute_inner)
