import ctypes
import json
import math
import re
import struct
from collections.abc import Sequence
from typing import Optional

from ._c_api import mta_string_t


_HEX_NUMBER = re.compile(r"(0[xX])?[0-9a-fA-F]+")


def _hex_from_cutoff(value: float) -> str:
    """
    Get the hexadecimal representation of the bit pattern of ``value``.

    Storing floating point values as their bit pattern makes the JSON
    round-trip exact, without relying on the precision of the decimal
    representation.
    """
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    return hex(bits)


def _cutoff_from_hex(value: str) -> float:
    """
    Inverse of :py:func:`_hex_from_cutoff`, reading a ``f64`` from the hexadecimal
    representation of its bit pattern.

    ``context`` is used to build the error message if ``value`` is not a valid
    hexadecimal string.
    """
    if isinstance(value, str) and _HEX_NUMBER.fullmatch(value) is not None:
        bits = int(value, 16)
    else:
        bits = 2**64

    if bits >= 2**64:
        raise ValueError(
            "'cutoff' in JSON for PairListOptions must be a hex-encoded string, "
            f"got '{value}'"
        )

    return struct.unpack("<d", struct.pack("<Q", bits))[0]


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


### ================================================================================ ###


class PairListOptions:
    """
    Options for the calculation of a pair list (also known as a neighbor list).

    A model declares the pair lists it needs with these options, and the engine running
    the model is then responsible for computing matching pair lists and attaching them
    to the systems given to the model.
    """

    def __init__(
        self,
        *,
        cutoff: float,
        full_list: bool,
        strict: bool = True,
        requestors: Optional[Sequence[str]] = None,
    ):
        """
        :param cutoff: spherical cutoff radius for this pair list, in the length unit of
            the model
        :param full_list: should the pair list be a full list (containing both the pair
            ``i -> j`` and ``j -> i``) or a half list (containing only ``i -> j``)
        :param strict: does the list only contain pairs within the cutoff (``True``) or
            can it also contain pairs slightly beyond the cutoff (``False``)
        :param requestors: list of strings describing who requested this pair list. More
            requestors can be added later with :py:meth:`add_requestor`.
        """
        self.cutoff = cutoff
        self.full_list = full_list
        self.strict = strict
        self.requestors = [] if requestors is None else requestors

    @property
    def cutoff(self) -> float:
        """
        Spherical cutoff radius for this pair list, in the length unit of the model.
        """
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: float):
        cutoff = float(value)

        if not math.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("cutoff must be a finite positive number")

        self._cutoff = cutoff

    @property
    def full_list(self) -> bool:
        """
        Should the pair list be a full list (containing both the pair ``i -> j`` and ``j
        -> i``) or a half list (containing only ``i -> j``)?
        """
        return self._full_list

    @full_list.setter
    def full_list(self, value: bool):
        self._full_list = bool(value)

    @property
    def strict(self) -> bool:
        """
        Does this list only contain pairs within the cutoff (``True``), or can it also
        contain pairs slightly beyond the cutoff (``False``)?
        """
        return self._strict

    @strict.setter
    def strict(self, value: bool):
        self._strict = bool(value)

    @property
    def requestors(self) -> list[str]:
        """
        List of strings describing who requested this pair list.

        The returned list is a copy, use :py:meth:`add_requestor` to register a new
        requestor.
        """
        return list(self._requestors)

    @requestors.setter
    def requestors(self, value: Sequence[str]):
        self._requestors = []
        for requestor in value:
            self.add_requestor(requestor)

    def add_requestor(self, requestor: str):
        """
        Add ``requestor`` to the list of entities requesting this pair list. Empty
        strings and duplicates are ignored.

        :param requestor: string describing who is requesting this pair list
        """
        requestor = str(requestor)
        if requestor != "" and requestor not in self._requestors:
            self._requestors.append(requestor)

    def to_dict(self) -> dict:
        """
        Convert this object to a JSON-compatible dictionary, following the
        :ref:`documented format <core-json-pair-options>`.
        """
        return {
            "type": "metatomic_pair_list_options",
            # store the bit pattern so the float round-trips exactly
            "cutoff": _hex_from_cutoff(self.cutoff),
            "full_list": self.full_list,
            "strict": self.strict,
            "requestors": self.requestors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PairListOptions":
        """
        Create a :py:class:`PairListOptions` from a JSON-compatible dictionary,
        following the :ref:`documented format <core-json-pair-options>`.

        :param data: dictionary containing the data, typically obtained by parsing JSON
            with :py:func:`json.loads`
        :raises ValueError: if the data does not match the expected format
        """
        if not isinstance(data, dict):
            raise ValueError(
                "invalid JSON data for PairListOptions, expected an object"
            )

        valid_keys = set(["type", "cutoff", "full_list", "strict", "requestors"])
        for key in data.keys():
            if key not in valid_keys:
                raise ValueError(f"unexpected key '{key}' in JSON for PairListOptions")

        if data.get("type") != "metatomic_pair_list_options":
            raise ValueError(
                "'type' in JSON for PairListOptions must be "
                "'metatomic_pair_list_options'"
            )

        cutoff = _cutoff_from_hex(data.get("cutoff"))
        if not math.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError(
                "'cutoff' in JSON for PairListOptions must be a finite positive number"
            )

        if not isinstance(data.get("full_list"), bool):
            raise ValueError(
                "'full_list' in JSON for PairListOptions must be a boolean"
            )

        if not isinstance(data.get("strict"), bool):
            raise ValueError("'strict' in JSON for PairListOptions must be a boolean")

        requestors = []
        if "requestors" in data:
            requestors = _check_string_list(
                data["requestors"], "'requestors' in JSON for PairListOptions"
            )

        return cls(
            cutoff=cutoff,
            full_list=data["full_list"],
            strict=data["strict"],
            requestors=requestors,
        )

    def __repr__(self) -> str:
        return (
            f"PairListOptions(cutoff={self._cutoff}, full_list={self._full_list}, "
            f"strict={self._strict})"
        )

    def _comparison_key(self):
        # the list of requestors is intentionally left out: two requests with
        # the same parameters can be fulfilled by the same pair list, whoever
        # asked for them
        return (self._cutoff, self._full_list, self._strict)

    # `PairListOptions` are compared by cutoff first, then `full_list` and
    # finally `strict`; the list of requestors is ignored everywhere.

    def __eq__(self, other) -> bool:
        """
        Check if two :py:class:`PairListOptions` are equal.

        The list of requestors is ignored when checking for equality.
        """
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() == other._comparison_key()

    def __ne__(self, other) -> bool:
        """
        Check if two :py:class:`PairListOptions` are different.

        The list of requestors is ignored when checking for equality.
        """
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() != other._comparison_key()

    def __lt__(self, other) -> bool:
        """Check if this pair list sorts before ``other``"""
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() < other._comparison_key()

    def __le__(self, other) -> bool:
        """Check if this pair list sorts before ``other`` or is equal to it"""
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() <= other._comparison_key()

    def __gt__(self, other) -> bool:
        """Check if this pair list sorts after ``other``"""
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() > other._comparison_key()

    def __ge__(self, other) -> bool:
        """Check if this pair list sorts after ``other`` or is equal to it"""
        if not isinstance(other, PairListOptions):
            return NotImplemented
        return self._comparison_key() >= other._comparison_key()

    def __hash__(self) -> int:
        return hash(self._comparison_key())
