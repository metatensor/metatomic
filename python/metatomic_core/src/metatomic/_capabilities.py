import math
from collections.abc import Sequence
from typing import Optional

from ._quantity import Quantity


_VALID_DTYPES = ["float32", "float64"]
_VALID_DEVICES = ["cpu", "cuda", "rocm", "metal"]


class ModelCapabilities:
    """
    Capabilities of a model: which outputs it can provide, which atomic types it
    supports, and other constraints on how it should be run.
    """

    def __init__(
        self,
        *,
        atomic_types: Sequence[int],
        interaction_range: float,
        length_unit: str,
        supported_devices: Sequence[str],
        dtype: str,
        outputs: Optional[Sequence[Quantity]] = None,
    ):
        """
        :param atomic_types: atomic types this model supports. The meaning of these
            integers is up to the model, and is not required to be the atomic numbers.
        :param interaction_range: interaction range of the model, in the length unit of
            the model
        :param length_unit: length unit of the model, e.g. ``"angstrom"``. All systems
            will be given to the model in this unit.
        :param supported_devices: devices on which the model can run, in order of
            preference
        :param dtype: data type of the model, used for all its inputs and outputs
        :param outputs: outputs this model can provide
        """
        self.atomic_types = atomic_types
        self.interaction_range = interaction_range
        self.length_unit = length_unit
        self.supported_devices = supported_devices
        self.dtype = dtype
        self.outputs = [] if outputs is None else outputs

    @property
    def outputs(self) -> list[Quantity]:
        """
        Outputs this model can provide.

        During a specific run, a model might be asked to only compute a subset of these
        outputs.

        The returned list is a copy, assign to this property to change the outputs.
        """
        return list(self._outputs)

    @outputs.setter
    def outputs(self, value: Sequence[Quantity]):
        outputs = []
        for output in value:
            if not isinstance(output, Quantity):
                raise ValueError(f"outputs must be Quantity, got {type(output)}")
            outputs.append(output)

        self._outputs = outputs

    @property
    def atomic_types(self) -> list[int]:
        """
        Atomic types this model supports.

        The meaning of the integers in this list is up to the model, and is not required
        to be the atomic numbers.

        The returned list is a copy, assign to this property to change the atomic types.
        """
        return list(self._atomic_types)

    @atomic_types.setter
    def atomic_types(self, value: Sequence[int]):
        atomic_types = []
        for atomic_type in value:
            if isinstance(atomic_type, bool) or not isinstance(atomic_type, int):
                raise ValueError(
                    f"atomic types must be integers, got {type(atomic_type)}"
                )
            atomic_types.append(atomic_type)

        self._atomic_types = atomic_types

    @property
    def interaction_range(self) -> float:
        """
        Interaction range of the model, in the length unit of the model.

        This is the maximum distance between two atoms for which the model's output can
        depend on their relative position. For a short range model, this is the same as
        the largest pair list cutoff; for a message passing model, this is the cutoff of
        one environment times the number of message passing steps; and for an explicit
        long range model, this should be set to infinity (``float("inf")``).
        """
        return self._interaction_range

    @interaction_range.setter
    def interaction_range(self, value: float):
        interaction_range = float(value)

        if math.isnan(interaction_range) or interaction_range < 0.0:
            raise ValueError("interaction_range must be non-negative")

        self._interaction_range = interaction_range

    @property
    def length_unit(self) -> str:
        """
        Length unit used by the model for its inputs, e.g. ``"angstrom"`` or
        ``"nanometer"``.

        This applies to the :py:attr:`interaction_range`, any cutoff in pair lists, the
        system positions, cell and pair lists given to the model.
        """
        return self._length_unit

    @length_unit.setter
    def length_unit(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"length_unit must be a string, got {type(value)}")

        self._length_unit = value

    @property
    def supported_devices(self) -> list[str]:
        """
        Devices on which this model can run.

        The devices should be ordered by preference: the first entry in this list should
        be the best device for this model, and so on.

        The returned list is a copy, assign to this property to change the supported
        devices.
        """
        return list(self._supported_devices)

    @supported_devices.setter
    def supported_devices(self, value: Sequence[str]):
        for device in value:
            if not isinstance(device, str):
                raise ValueError(
                    f"devices must be a list of strings, got {type(device)}"
                )
            if device not in _VALID_DEVICES:
                raise ValueError(
                    f"device must be one of {list(_VALID_DEVICES)}, got {device}"
                )

        self._supported_devices = list(value)

    @property
    def dtype(self) -> str:
        """
        Data type of this model, used for all its inputs and outputs.

        The model is free to use a different data type for its internal computations.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"dtype must be a string, got {type(value)}")
        if value not in _VALID_DTYPES:
            raise ValueError(f"dtype must be one of {list(_VALID_DTYPES)}, got {value}")

        self._dtype = value

    def find_output(self, request: Quantity) -> Optional[Quantity]:
        """
        Find the output matching the name and sample kind of ``request``, or ``None`` if
        this model does not declare such an output.

        :param request: the quantity to look for
        """
        for output in self._outputs:
            if (
                output.name == request.name
                and output.sample_kind == request.sample_kind
            ):
                return output

        return None

    def to_dict(self) -> dict:
        """
        Convert this object to a JSON-compatible dictionary, following the
        :ref:`documented format <core-json-model-capabilities>`.
        """
        return {
            "type": "metatomic_model_capabilities",
            "outputs": [output.to_dict() for output in self._outputs],
            "atomic_types": self.atomic_types,
            "interaction_range": self.interaction_range,
            "length_unit": self.length_unit,
            "supported_devices": self._supported_devices,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelCapabilities":
        """
        Create a :py:class:`ModelCapabilities` from a JSON-compatible dictionary,
        following the :ref:`documented format <core-json-model-capabilities>`.

        :param data: dictionary containing the data, typically obtained by parsing JSON
            with :py:func:`json.loads`
        :raises ValueError: if the data does not match the expected format
        """
        if not isinstance(data, dict):
            raise ValueError(
                "invalid JSON data for ModelCapabilities, expected an object"
            )

        valid_keys = set(
            [
                "type",
                "outputs",
                "atomic_types",
                "interaction_range",
                "length_unit",
                "supported_devices",
                "dtype",
            ]
        )
        for key in data.keys():
            if key not in valid_keys:
                raise ValueError(
                    f"unexpected key '{key}' in JSON for ModelCapabilities"
                )

        if data.get("type") != "metatomic_model_capabilities":
            raise ValueError(
                "'type' in JSON for ModelCapabilities must be "
                "'metatomic_model_capabilities'"
            )

        if not isinstance(data.get("outputs"), list):
            raise ValueError("'outputs' in JSON for ModelCapabilities must be an array")
        outputs = [Quantity.from_dict(output) for output in data["outputs"]]

        if not isinstance(data.get("atomic_types"), list):
            raise ValueError(
                "'atomic_types' in JSON for ModelCapabilities must be an array"
            )

        for atomic_type in data["atomic_types"]:
            if isinstance(atomic_type, bool) or not isinstance(atomic_type, int):
                raise ValueError(
                    "'atomic_types' in JSON for ModelCapabilities must be an "
                    "array of integers"
                )

        interaction_range = data.get("interaction_range")
        if isinstance(interaction_range, bool) or not isinstance(
            interaction_range, (int, float)
        ):
            raise ValueError(
                "'interaction_range' in JSON for ModelCapabilities must be a number"
            )
        if interaction_range < 0.0:
            raise ValueError(
                "'interaction_range' in JSON for ModelCapabilities must be non-negative"
            )

        if not isinstance(data.get("length_unit"), str):
            raise ValueError(
                "'length_unit' in JSON for ModelCapabilities must be a string"
            )

        if not isinstance(data.get("supported_devices"), list):
            raise ValueError(
                "'supported_devices' in JSON for ModelCapabilities must be an array"
            )

        if not isinstance(data.get("dtype"), str):
            raise ValueError("dtype in JSON for ModelCapabilities must be a string")

        return cls(
            atomic_types=data["atomic_types"],
            interaction_range=interaction_range,
            length_unit=data["length_unit"],
            supported_devices=data["supported_devices"],
            dtype=data["dtype"],
            outputs=outputs,
        )

    def __repr__(self) -> str:
        return (
            f"ModelCapabilities(outputs={self._outputs!r}, "
            f"atomic_types={self._atomic_types}, "
            f"interaction_range={self._interaction_range}, "
            f"length_unit='{self._length_unit}', "
            f"supported_devices={[str(d) for d in self._supported_devices]}, "
            f"dtype='{self._dtype}')"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModelCapabilities):
            return NotImplemented

        return (
            self._outputs == other._outputs
            and self._atomic_types == other._atomic_types
            and self._interaction_range == other._interaction_range
            and self._length_unit == other._length_unit
            and self._supported_devices == other._supported_devices
            and self._dtype == other._dtype
        )
