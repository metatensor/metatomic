from collections.abc import Sequence
from typing import Optional


_VALID_SAMPLE_KINDS = ["system", "atom", "atom_pair"]
_VALID_GRADIENTS = ["positions", "strain"]


class Quantity:
    """
    A physical quantity that a model can take as input or produce as output.
    """

    def __init__(
        self,
        *,
        name: str,
        unit: str,
        sample_kind: str,
        description: Optional[str] = None,
        gradients: Optional[Sequence[str]] = None,
    ):
        """
        :param name: name of the quantity, either one of the :ref:`standard names
            <standard-quantities>` or a custom name of the form
            ``<namespace>::<name>[/<variant>]``.
        :param unit: unit of the quantity
        :param sample_kind: kind of samples this quantity is associated with
        :param description: optional description of this quantity, especially useful
            when a model defines multiple variants of the same quantity
        :param gradients: list of gradients stored explicitly in the
            :py:class:`TensorMap <metatensor.TensorMap>` for this quantity
        """
        self.name = name
        self.unit = unit
        self.sample_kind = sample_kind
        self.description = description
        self.gradients = [] if gradients is None else gradients

    @property
    def name(self) -> str:
        """
        Name of this quantity, this can be one of the :ref:`standard names
        <standard-quantities>` or a custom name of the form
        ``<namespace>::<name>[/<variant>]``.

        This is not validated here: an invalid name will be rejected by the shared
        library when the quantity is sent to it.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"name must be a string, got {type(value)}")
        self._name = value

    @property
    def unit(self) -> str:
        """Unit of this quantity"""
        return self._unit

    @unit.setter
    def unit(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"unit must be a string, got {type(value)}")
        self._unit = value

    @property
    def sample_kind(self) -> str:
        """
        Kind of samples this quantity is associated with. This is can be one of the
        following:

        - ``system`` for per-system/global quantity
        - ``atom`` for per-atom quantity
        - ``atom_pair`` for quantities defined over a pair of atoms
        """
        return self._sample_kind

    @sample_kind.setter
    def sample_kind(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"sample_kind must be a string, got {type(value)}")

        if value not in _VALID_SAMPLE_KINDS:
            raise ValueError(
                f"sample_kind must be one of {list(_VALID_SAMPLE_KINDS)}, got {value}"
            )

        self._sample_kind = value

    @property
    def description(self) -> Optional[str]:
        """
        Description of this quantity, used to provide more details about it, especially
        when a model defines multiple variants of the same quantity.
        """
        return self._description

    @description.setter
    def description(self, value: Optional[str]):
        if value is None or value == "":
            # an empty description is the same as no description at all
            self._description = None
        elif isinstance(value, str):
            self._description = value
        else:
            raise ValueError(f"description must be a string, got {type(value)}")

    @property
    def gradients(self) -> list[str]:
        """
        List of gradients stored explicitly in the :py:class:`TensorMap
        <metatensor.TensorMap>` for this quantity.

        Gradients can be one of the following:

        - ``positions`` for gradients with respect to positions, e.g. the forces
        - ``strain`` for gradients with respect to the strain, e.g. the stress tensor

        The returned list is a copy, assign to this property to change the gradients.
        """
        return list(self._gradients)

    @gradients.setter
    def gradients(self, value: Sequence[str]):
        for gradient in value:
            if not isinstance(gradient, str):
                raise ValueError(
                    f"gradients must be a list of strings, got {type(gradient)}"
                )
            if gradient not in _VALID_GRADIENTS:
                raise ValueError(
                    f"gradient must be one of {list(_VALID_GRADIENTS)}, got {gradient}"
                )

        self._gradients = list(value)

    def to_dict(self) -> dict:
        """
        Convert this object to a JSON-compatible dictionary, following the
        :ref:`documented format <core-json-quantity>`.
        """
        result = {
            "type": "metatomic_quantity",
            "name": self.name,
            "unit": self.unit,
            "sample_kind": self.sample_kind,
            "gradients": self._gradients,
        }

        if self.description is not None:
            result["description"] = self.description

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Quantity":
        """
        Create a :py:class:`Quantity` from a JSON-compatible dictionary, following the
        :ref:`documented format <core-json-quantity>`.

        :param data: dictionary containing the data, typically obtained by parsing JSON
        """
        if not isinstance(data, dict):
            raise ValueError("invalid JSON data for Quantity, expected an object")

        valid_keys = set(
            ["type", "name", "unit", "sample_kind", "description", "gradients"]
        )
        for key in data.keys():
            if key not in valid_keys:
                raise ValueError(f"unexpected key '{key}' in JSON for Quantity")

        if data.get("type") != "metatomic_quantity":
            raise ValueError("'type' in JSON for Quantity must be 'metatomic_quantity'")

        if not isinstance(data.get("name"), str):
            raise ValueError("'name' in JSON for Quantity must be a string")

        if not isinstance(data.get("unit"), str):
            raise ValueError("'unit' in JSON for Quantity must be a string")

        description = data.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError("'description' in JSON for Quantity must be a string")

        if not isinstance(data.get("gradients"), list):
            raise ValueError("'gradients' in JSON for Quantity must be an array")

        if not isinstance(data.get("sample_kind"), str):
            raise ValueError("'sample_kind' in JSON for Quantity must be a string")

        return cls(
            name=data["name"],
            unit=data["unit"],
            sample_kind=data["sample_kind"],
            description=description,
            gradients=data["gradients"],
        )

    def __repr__(self) -> str:
        return (
            f"Quantity(name='{self._name}', unit='{self._unit}', "
            f"sample_kind='{self._sample_kind}', description={self._description!r}, "
            f"gradients={[str(g) for g in self._gradients]})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented

        return (
            self._name == other._name
            and self._unit == other._unit
            and self._sample_kind == other._sample_kind
            and self._description == other._description
            and self._gradients == other._gradients
        )
