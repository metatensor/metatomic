import ctypes
import json
from collections.abc import Mapping, Sequence
from typing import Optional, Union

from ._c_api import mta_string_t


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
    try:
        return lib.mta_string_view(printed).decode("utf8")
    finally:
        lib.mta_string_free(printed)


def _check_string_list(values, context: str) -> list[str]:
    """
    Check that ``values`` is a list of strings, and return it as a new
    :py:class:`list`. ``context`` is used to build the error messages.
    """
    if not isinstance(values, list):
        raise ValueError(f"{context} must be an array")

    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{context} must be an array of strings")

    return list(values)


class References:
    """
    References for a model, divided into three categories: references about the model as
    a whole, references about the architecture of the model, and references about the
    implementation of the model.

    Each category is a list of strings, which can be DOIs, URLs, or any other format the
    model author finds useful.
    """

    def __init__(
        self,
        *,
        model: Optional[Sequence[str]] = None,
        architecture: Optional[Sequence[str]] = None,
        implementation: Optional[Sequence[str]] = None,
    ):
        """
        :param model: references about the model as a whole, e.g. a paper describing the
            model or a website presenting it
        :param architecture: references about the architecture of the model, e.g. papers
            describing the mathematical form of the model
        :param implementation: references about the implementation of the model, e.g. a
            link to the source code repository or a paper describing the software
        """
        self.model = [] if model is None else model
        self.architecture = [] if architecture is None else architecture
        self.implementation = [] if implementation is None else implementation

    @staticmethod
    def _check_section(values: Sequence[str], section: str) -> list[str]:
        result = []
        for value in values:
            if not isinstance(value, str):
                raise ValueError(
                    f"reference must be a string (in '{section}' section), "
                    f"got {type(value)}"
                )
            if value == "":
                raise ValueError(
                    f"reference can not be empty string (in '{section}' section)"
                )
            result.append(value)

        return result

    @property
    def model(self) -> list[str]:
        """
        References about the model as a whole, e.g. a paper describing the model or a
        website presenting it.

        The returned list is a copy, use :py:meth:`add` to register a new reference.
        """
        return list(self._model)

    @model.setter
    def model(self, value: Sequence[str]):
        self._model = self._check_section(value, "model")

    @property
    def architecture(self) -> list[str]:
        """
        References about the architecture of the model, e.g. papers describing the
        mathematical form of the model.

        The returned list is a copy, use :py:meth:`add` to register a new reference.
        """
        return list(self._architecture)

    @architecture.setter
    def architecture(self, value: Sequence[str]):
        self._architecture = self._check_section(value, "architecture")

    @property
    def implementation(self) -> list[str]:
        """
        References about the implementation of the model, e.g. a link to the source code
        repository or a paper describing the software.

        The returned list is a copy, use :py:meth:`add` to register a new reference.
        """
        return list(self._implementation)

    @implementation.setter
    def implementation(self, value: Sequence[str]):
        self._implementation = self._check_section(value, "implementation")

    def add(self, section: str, reference: str):
        """
        Add ``reference`` to the given ``section``.

        :param section: one of ``"model"``, ``"architecture"``, or ``"implementation"``
        :param reference: the reference to add
        """
        if section not in ["model", "architecture", "implementation"]:
            raise ValueError(
                "reference section must be 'model', 'architecture', or "
                f"'implementation', got '{section}'"
            )

        checked = self._check_section([reference], section)
        getattr(self, f"_{section}").extend(checked)

    def to_dict(self) -> dict:
        """Convert this object to a JSON-compatible dictionary"""
        return {
            "model": self.model,
            "architecture": self.architecture,
            "implementation": self.implementation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "References":
        """
        Create a :py:class:`References` from a JSON-compatible dictionary.

        :param data: dictionary containing the data, typically obtained by parsing JSON
            with :py:func:`json.loads`
        :raises ValueError: if the data does not match the expected format
        """
        if not isinstance(data, dict):
            raise ValueError(
                "invalid JSON data for references in ModelMetadata, expected an object"
            )

        valid_keys = set(["model", "architecture", "implementation"])
        for key in data.keys():
            if key not in valid_keys:
                raise ValueError(f"unexpected key '{key}' in JSON for references")

        return cls(
            model=_check_string_list(
                data.get("model"), "'model' in references of ModelMetadata"
            ),
            architecture=_check_string_list(
                data.get("architecture"),
                "'architecture' in references of ModelMetadata",
            ),
            implementation=_check_string_list(
                data.get("implementation"),
                "'implementation' in references of ModelMetadata",
            ),
        )

    def __repr__(self) -> str:
        return (
            f"References(model={self._model}, architecture={self._architecture}, "
            f"implementation={self._implementation})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, References):
            return NotImplemented

        return (
            self._model == other._model
            and self._architecture == other._architecture
            and self._implementation == other._implementation
        )


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
        references: Union["References", Mapping[str, Sequence[str]], None] = None,
        extra: Optional[Mapping[str, str]] = None,
    ):
        """
        :param name: name of the model, e.g. ``"MyCoolModel v1.2"``
        :param description: free-text description of the model
        :param authors: authors of the model, e.g. ``["Alice Smith", "Bob Johnson
            <bobj@example.com>"]``
        :param references: references for the model that should be cited when using it,
            either a :py:class:`References` or a dictionary with ``"model"``,
            ``"architecture"`` and ``"implementation"`` keys
        :param extra: any other key-value pairs the model author wants to include in the
            metadata. This can be used for any purpose.
        """
        self.name = name
        self.description = description
        self.authors = [] if authors is None else authors
        self.references = References() if references is None else references
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

        The returned list is a copy, use :py:meth:`add_author` to register a new author.
        """
        return list(self._authors)

    @authors.setter
    def authors(self, value: Sequence[str]):
        self._authors = []
        for author in value:
            self.add_author(author)

    def add_author(self, author: str):
        """
        Add ``author`` to the list of authors of this model.

        :param author: name of the author, optionally followed by an email address
            between angle brackets
        """
        if not isinstance(author, str):
            raise ValueError(f"author must be a string, got {type(author)}")

        if author == "":
            raise ValueError("author can not be empty string in ModelMetadata")

        self._authors.append(author)

    @property
    def references(self) -> References:
        """References for the model that should be cited when using it"""
        return self._references

    @references.setter
    def references(self, value: Union[References, Mapping[str, Sequence[str]]]):
        if isinstance(value, References):
            self._references = value
        elif isinstance(value, Mapping):
            unknown = set(value.keys()) - {"model", "architecture", "implementation"}
            if unknown:
                raise ValueError(
                    "reference section must be 'model', 'architecture', or "
                    f"'implementation', got '{sorted(unknown)[0]}'"
                )
            self._references = References(**value)
        else:
            raise ValueError(
                f"references must be a References or a dict, got {type(value)}"
            )

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
            "references": self.references.to_dict(),
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

        references = References.from_dict(data.get("references"))

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
