import copy
import math
import operator

import pytest

from metatomic import PairListOptions


### ================================================================================ ###
###                                  PairListOptions                                 ###
### ================================================================================ ###


@pytest.fixture
def pair_options():
    return PairListOptions(
        cutoff=3.5,
        full_list=True,
        strict=False,
        requestors=["nl-1", "nl-2"],
    )


def test_pair_options(pair_options):
    assert pair_options.strict is False
    assert pair_options.requestors == ["nl-1", "nl-2"]

    # setters & getters
    pair_options.cutoff = 1.0
    pair_options.full_list = False
    pair_options.strict = True
    pair_options.requestors = ["foo"]
    assert pair_options.cutoff == 1.0
    assert pair_options.full_list is False
    assert pair_options.strict is True
    assert pair_options.requestors == ["foo"]

    defaults = PairListOptions(cutoff=3.5, full_list=True)
    assert defaults.cutoff == 3.5
    assert defaults.full_list is True
    # `strict` defaults to True
    assert defaults.strict is True
    assert defaults.requestors == []


def test_pair_options_invalid_cutoff():
    for cutoff in [0.0, -1.0, math.inf, -math.inf, math.nan]:
        message = "cutoff must be a finite positive number"
        with pytest.raises(ValueError, match=message):
            PairListOptions(cutoff=cutoff, full_list=True)


def test_pair_options_requestors(pair_options):
    options = PairListOptions(cutoff=3.5, full_list=True)

    options.add_requestor("nl-1")
    options.add_requestor("nl-2")
    # empty strings and duplicates are ignored, first-seen order is preserved
    options.add_requestor("nl-1")
    options.add_requestor("")
    assert options.requestors == ["nl-1", "nl-2"]

    # the returned list is a copy
    options.requestors.append("nl-3")
    assert options.requestors == ["nl-1", "nl-2"]

    options.requestors = ["a", "", "b", "a"]
    assert options.requestors == ["a", "b"]

    # handle duplicate/empty strings in from_dict
    data = pair_options.to_dict()
    data["requestors"] = ["a", "", "b", "a"]

    parsed = PairListOptions.from_dict(data)
    assert parsed.requestors == ["a", "b"]


def test_pair_options_comparison(pair_options):
    # the requestors are ignored when comparing
    other = copy.deepcopy(pair_options)
    other.add_requestor("nl-3")
    assert pair_options == other
    assert hash(pair_options) == hash(other)

    other = copy.deepcopy(pair_options)
    other.cutoff = 4.0
    assert pair_options != other
    assert pair_options < other
    assert pair_options <= other
    assert other > pair_options
    assert other >= pair_options

    other = copy.deepcopy(pair_options)
    other.strict = True
    assert pair_options != other
    assert pair_options < other

    # a pair list compares equal to (and neither before nor after) itself
    same = copy.deepcopy(pair_options)
    assert pair_options <= same
    assert pair_options >= same
    assert not pair_options < same
    assert not pair_options > same

    assert pair_options != "not a PairListOptions"
    for op in [operator.lt, operator.le, operator.gt, operator.ge]:
        with pytest.raises(TypeError):
            op(pair_options, "not a PairListOptions")

    unsorted = [
        PairListOptions(cutoff=4.0, full_list=False),
        PairListOptions(cutoff=1.0, full_list=True),
        PairListOptions(cutoff=1.0, full_list=False),
    ]
    assert sorted(unsorted) == [unsorted[2], unsorted[1], unsorted[0]]


def test_pair_options_roundtrip(pair_options):
    data = pair_options.to_dict()

    assert data == {
        "type": "metatomic_pair_list_options",
        "cutoff": "0x400c000000000000",
        "full_list": True,
        "strict": False,
        "requestors": ["nl-1", "nl-2"],
    }

    parsed = PairListOptions.from_dict(data)
    assert parsed == pair_options
    assert parsed.requestors == pair_options.requestors


def test_pair_options_cutoff_keeps_full_precision():
    options = PairListOptions(cutoff=1.0 / 3.0, full_list=True)
    parsed = PairListOptions.from_dict(options.to_dict())
    assert parsed.cutoff == options.cutoff


def test_pair_options_from_dict_errors(pair_options):
    def corrupted(**kwargs):
        data = pair_options.to_dict()
        data.update(kwargs)
        return data

    def without(key):
        data = pair_options.to_dict()
        del data[key]
        return data

    # each case corrupts exactly one field of an otherwise valid object
    cases = [
        (
            "not an object",
            "invalid JSON data for PairListOptions, expected an object",
        ),
        (
            corrupted(type="something-else"),
            "'type' in JSON for PairListOptions must be 'metatomic_pair_list_options'",
        ),
        (
            without("cutoff"),
            "'cutoff' in JSON for PairListOptions must be a hex-encoded string",
        ),
        (
            corrupted(cutoff="not-hex"),
            "'cutoff' in JSON for PairListOptions must be a hex-encoded string",
        ),
        (
            corrupted(cutoff=3.5),
            "'cutoff' in JSON for PairListOptions must be a hex-encoded string",
        ),
        (
            corrupted(cutoff="0x7ff8000000000000"),  # NaN
            "'cutoff' in JSON for PairListOptions must be a finite positive number",
        ),
        (
            corrupted(cutoff="0x7ff0000000000000"),  # +inf
            "'cutoff' in JSON for PairListOptions must be a finite positive number",
        ),
        (
            corrupted(cutoff="0xbff0000000000000"),  # -1.0
            "'cutoff' in JSON for PairListOptions must be a finite positive number",
        ),
        (
            corrupted(cutoff="0x0"),  # 0.0
            "'cutoff' in JSON for PairListOptions must be a finite positive number",
        ),
        (
            corrupted(full_list="yes"),
            "'full_list' in JSON for PairListOptions must be a boolean",
        ),
        (
            without("strict"),
            "'strict' in JSON for PairListOptions must be a boolean",
        ),
        (
            corrupted(requestors="nl-1"),
            "'requestors' in JSON for PairListOptions must be an array",
        ),
        (
            corrupted(requestors=["nl-1", 42]),
            "'requestors' in JSON for PairListOptions must be an array of strings",
        ),
    ]

    for data, message in cases:
        with pytest.raises(ValueError, match=message):
            PairListOptions.from_dict(data)
