import copy
import ctypes
import json
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional

from ._c_api import mta_string_t
from ._utils import _string_from_mta


def _format_metadata(metadata: dict) -> str:
    """
    Call ``mta_format_metadata`` to render a JSON-serialized
    :py:class:`ModelMetadata` as human-readable text.

    The formatting is done by the shared library to make sure all the languages
    supported by metatomic produce exactly the same output.
    """
    from ._c_lib import _get_library

    lib = _get_library()

    printed = mta_string_t()
    lib.mta_format_metadata(json.dumps(metadata).encode("utf8"), ctypes.byref(printed))
    return _string_from_mta(printed)


def _check_string_list(values, context: str) -> list[str]:
    """
    Check that ``values`` is a list of strings, and return it as a new :py:class:`list`.
    ``context`` is used to build the error messages.

    If ``allow_empty`` is ``True``, empty strings are allowed in the list.
    """
    if not isinstance(values, list):
        raise ValueError(f"{context} must be an array")

    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{context} must be an array of strings")

        if value == "":
            raise ValueError(f"{context} cannot contain empty strings")

    return list(values)


class ModelMetadata:
    """
    Metadata about a model: who created it, what it does, and which references should be
    cited when using it.

    This class implements ``__str__``, so the metadata can be pretty-printed with
    ``print(metadata)``.
    """

    def __init__(
        self,
        *,
        name: str = "",
        description: str = "",
        authors: Optional[Sequence[str]] = None,
        references: Optional[Mapping[str, Sequence[str]]] = None,
        extra: Optional[Mapping[str, str]] = None,
    ):
        """
        :param name: name of the model, e.g. ``"MyCoolModel v1.2"``
        :param description: free-text description of the model
        :param authors: authors of the model, e.g. ``["Alice Smith", "Bob Johnson
            <bobj@example.com>"]``
        :param references: references for the model that should be cited when using it,
            as a dictionary with ``"model"``, ``"architecture"`` and
            ``"implementation"`` keys
        :param extra: any other key-value pairs the model author wants to include in the
            metadata. This can be used for any purpose.
        """
        self.name = name
        self.description = description
        self.authors = [] if authors is None else authors
        self.references = {} if references is None else references
        self.extra = {} if extra is None else extra

    @property
    def name(self) -> str:
        """Name of the model, e.g. ``"MyCoolModel v1.2"``"""
        return self._name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"name must be a string, got {type(value)}")
        self._name = value

    @property
    def description(self) -> str:
        """Free-text description of the model"""
        return self._description

    @description.setter
    def description(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"description must be a string, got {type(value)}")
        self._description = value

    @property
    def authors(self) -> list[str]:
        """
        Authors of the model, e.g. ``["Alice Smith", "Bob Johnson
        <bobj@example.com>"]``.

        The returned list is a copy, assign to the property to change the whole list of
        authors.
        """
        return list(self._authors)

    @authors.setter
    def authors(self, value: Sequence[str]):
        self._authors = []
        for author in value:
            if not isinstance(author, str):
                raise ValueError(f"author must be a string, got {type(author)}")

            if author == "":
                raise ValueError("author can not be empty string in ModelMetadata")

            self._authors.append(author)

    @property
    def references(self) -> Dict[str, List[str]]:
        """References for the model that should be cited when using it"""
        return copy.deepcopy(self._references)

    @references.setter
    def references(self, value: Mapping[str, Sequence[str]]):
        self._references = {}
        if isinstance(value, Mapping):
            unknown = set(value.keys()) - {"model", "architecture", "implementation"}
            if unknown:
                raise ValueError(
                    "reference section must be 'model', 'architecture', or "
                    f"'implementation', got '{sorted(unknown)[0]}'"
                )

            self._references["model"] = _check_string_list(
                value.get("model", []), "model references"
            )
            self._references["architecture"] = _check_string_list(
                value.get("architecture", []), "architecture references"
            )
            self._references["implementation"] = _check_string_list(
                value.get("implementation", []), "implementation references"
            )
        else:
            raise ValueError(f"references must be a dict, got {type(value)}")

    @property
    def extra(self) -> dict[str, str]:
        """
        Any other key-value pairs the model author wants to include in the metadata.
        This can be used for any purpose.

        The returned dictionary is a copy, assign to this property to change the extra
        metadata.
        """
        return dict(self._extra)

    @extra.setter
    def extra(self, value: Mapping[str, str]):
        extra = {}
        for key, entry in value.items():
            if not isinstance(key, str):
                raise ValueError(f"extra keys must be strings, got {type(key)}")
            if not isinstance(entry, str):
                raise ValueError(f"extra values must be strings, got {type(entry)}")
            extra[key] = entry

        self._extra = extra

    def to_dict(self) -> dict:
        """
        Convert this object to a JSON-compatible dictionary, following the
        :ref:`documented format <core-json-model-metadata>`.
        """
        return {
            "type": "metatomic_model_metadata",
            "name": self.name,
            "authors": self.authors,
            "description": self.description,
            "references": self.references,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        """
        Create a :py:class:`ModelMetadata` from a JSON-compatible dictionary, following
        the :ref:`documented format <core-json-model-metadata>`.

        :param data: dictionary containing the data, typically obtained by parsing JSON
            with :py:func:`json.loads`
        :raises ValueError: if the data does not match the expected format
        """
        if not isinstance(data, dict):
            raise ValueError("invalid JSON data for ModelMetadata, expected an object")

        valid_keys = set(
            ["type", "name", "authors", "description", "references", "extra"]
        )
        for key in data.keys():
            if key not in valid_keys:
                raise ValueError(f"unexpected key '{key}' in JSON for ModelMetadata")

        if data.get("type") != "metatomic_model_metadata":
            raise ValueError(
                "'type' in JSON for ModelMetadata must be 'metatomic_model_metadata'"
            )

        if not isinstance(data.get("name"), str):
            raise ValueError("'name' in JSON for ModelMetadata must be a string")

        authors = _check_string_list(
            data.get("authors"), "'authors' in JSON for ModelMetadata"
        )

        if not isinstance(data.get("description"), str):
            raise ValueError("'description' in JSON for ModelMetadata must be a string")

        reference_data = data.get("references")
        if not isinstance(reference_data, dict):
            raise ValueError(
                "invalid JSON data for references in ModelMetadata, expected an object"
            )

        valid_keys = set(["model", "architecture", "implementation"])
        for key in reference_data.keys():
            if key not in valid_keys:
                raise ValueError(f"unexpected key '{key}' in JSON for references")

        references = {
            "model": _check_string_list(
                reference_data.get("model"), "'model' in references of ModelMetadata"
            ),
            "architecture": _check_string_list(
                reference_data.get("architecture"),
                "'architecture' in references of ModelMetadata",
            ),
            "implementation": _check_string_list(
                reference_data.get("implementation"),
                "'implementation' in references of ModelMetadata",
            ),
        }

        if not isinstance(data.get("extra"), dict):
            raise ValueError("'extra' in JSON for ModelMetadata must be an object")

        for value in data["extra"].values():
            if not isinstance(value, str):
                raise ValueError(
                    "'extra' in JSON for ModelMetadata must be an object with "
                    "string values"
                )

        return cls(
            name=data["name"],
            description=data["description"],
            authors=authors,
            references=references,
            extra=data["extra"],
        )

    def __str__(self) -> str:
        return _format_metadata(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"ModelMetadata(name='{self._name}', description='{self._description}', "
            f"authors={self._authors}, references={self._references!r}, "
            f"extra={self._extra})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModelMetadata):
            return NotImplemented

        return (
            self._name == other._name
            and self._description == other._description
            and self._authors == other._authors
            and self._references == other._references
            and self._extra == other._extra
        )
