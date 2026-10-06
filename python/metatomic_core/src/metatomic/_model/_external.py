import ctypes
import json
from typing import Optional

from metatensor import Labels, TensorMap

from .._c_api import mta_model_t, mta_string_t
from .._c_lib import _get_library
from .._capabilities import ModelCapabilities
from .._metadata import ModelMetadata
from .._quantity import Quantity
from .._status import check_status
from .._system import PairListOptions, System
from .._utils import _string_from_mta
from ._base import _CALLBACK_TYPES, BaseModel


class ExternalModel(BaseModel):
    """
    Wrapper around an existing ``mta_model_t``, for example a model loaded from a
    plugin.

    This exposes the model through the same :py:class:`BaseModel` interface. The
    :py:class:`ExternalModel` owns the underlying ``mta_model_t``, and calls its
    ``unload`` callback when garbage-collected.
    """

    def __init__(self, model: mta_model_t):
        """
        :param model: model to wrap. The :py:class:`ExternalModel` takes ownership of
            this model.
        """
        if not isinstance(model, mta_model_t):
            raise TypeError(f"`model` must be a mta_model_t, not {type(model)}")

        self._lib = _get_library()
        self._model = model

    def __del__(self):
        model = getattr(self, "_model", None)
        if model is not None and model.unload:
            self._model = None
            check_status(model.unload(model.data))

    def _check_callback(self, name: str):
        if not getattr(self._model, name):
            raise ValueError(f"model is missing a '{name}' callback")

    def _call_json_callback(self, name: str):
        self._check_callback(name)

        output = mta_string_t()
        check_status(getattr(self._model, name)(self._model.data, ctypes.byref(output)))
        return json.loads(_string_from_mta(output))

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities.from_dict(self._call_json_callback("capabilities"))

    def metadata(self) -> ModelMetadata:
        return ModelMetadata.from_dict(self._call_json_callback("metadata"))

    def requested_pair_lists(self) -> list[PairListOptions]:
        data = self._call_json_callback("requested_pair_lists")
        return [PairListOptions.from_dict(item) for item in data]

    def requested_inputs(self) -> list[Quantity]:
        data = self._call_json_callback("requested_inputs")
        return [Quantity.from_dict(item) for item in data]

    def execute_inner(
        self,
        systems: list[System],
        selected_atoms: Optional[Labels],
        requested_outputs: list[Quantity],
    ) -> list[TensorMap]:
        raise RuntimeError(
            "ExternalModel.execute_inner() should never be called directly. "
            "Use execute_model() instead."
        )

    def _as_mta_model_t(self) -> mta_model_t:
        # Replace _as_mta_model_t from BaseModel to prevent double wrapping of the
        # model. The returned mta_model_t is a view of the underlying mta_model_t, and
        # does not take ownership of it.
        model = mta_model_t.from_buffer_copy(self._model)
        model.unload = _CALLBACK_TYPES["unload"]()
        return model
