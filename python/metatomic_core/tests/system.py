import copy
import json
import math
import operator

import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import MetatomicError, PairListOptions, System


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

    # a bare string is a Sequence[str] but would silently split into characters
    options.requestors = ["keep-me"]
    with pytest.raises(TypeError, match="sequence of strings, not a single string"):
        options.requestors = "engine"
    assert options.requestors == ["keep-me"]

    options.requestors = ("a", "b")
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


### ================================================================================ ###
###                                      System                                      ###
### ================================================================================ ###


@pytest.fixture
def system():
    n_atoms = 4
    types = np.array([i * 3 + 1 for i in range(n_atoms)], dtype=np.int32)
    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        positions[i] = (i * 3 + 1, i * 3 + 2, i * 3 + 3)
    cell = np.array(
        [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=np.float64,
    )
    pbc = np.array([True, False, True])
    return System("nm", types, positions, cell, pbc)


@pytest.fixture
def pair_block():
    return TensorBlock(
        values=np.array([[[1.5], [2.5], [3.5]]], dtype=np.float64),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            np.array([[0, 1, 0, 0, 0]], dtype=np.int32),
        ),
        components=[Labels("xyz", np.array([[0], [1], [2]], dtype=np.int32))],
        properties=Labels("distance", np.array([[0]], dtype=np.int32)),
    )


@pytest.fixture
def custom_data():
    block = TensorBlock(
        values=np.array([[42.0]], dtype=np.float64),
        samples=Labels("sample", np.array([[0]], dtype=np.int32)),
        components=[],
        properties=Labels("property", np.array([[0]], dtype=np.int32)),
    )
    return TensorMap(Labels("key", np.array([[0]], dtype=np.int32)), [block])


def test_system_basics(system):
    assert system.size == 4
    assert len(system) == 4
    assert system.length_unit == "nm"
    assert system.arrays_backend == "numpy"
    assert isinstance(system.positions, np.ndarray)


def test_system_construction_errors():
    types = np.array([1, 2, 3], dtype=np.float32)
    positions = np.zeros((3, 3), dtype=np.float32)
    cell = np.eye(3, dtype=np.float32)
    pbc = np.array([True, True, True])
    with pytest.raises(MetatomicError, match="types"):
        System("Angstrom", types, positions, cell, pbc)


def test_system_data(system):
    types = np.asarray(system.types)
    assert types.shape == (4,)
    assert not types.flags.writeable
    assert types[0] == 1
    assert types[3] == 10

    positions = np.asarray(system.positions)
    assert positions.shape == (4, 3)
    assert not positions.flags.writeable
    assert positions[0, 0] == 1.0
    assert positions[3, 0] == 10.0

    cell = np.asarray(system.cell)
    assert cell.shape == (3, 3)
    assert not cell.flags.writeable

    pbc = np.asarray(system.pbc)
    assert pbc.shape == (3,)
    assert not pbc.flags.writeable
    assert bool(pbc[0]) is True
    assert bool(pbc[1]) is False
    assert bool(pbc[2]) is True

    # DLPack views keep the backing system storage alive.
    del system
    assert positions[3, 0] == 10.0


def test_system_pairs(system, pair_block):
    options = PairListOptions(
        cutoff=1.0, full_list=True, strict=False, requestors=["test"]
    )
    system.add_pairs(options, pair_block.copy())

    options_json = json.dumps(
        {
            "type": "metatomic_pair_list_options",
            "cutoff": "0x40364ccccccccccd",
            "full_list": False,
            "strict": True,
            "requestors": [""],
        }
    )
    system.add_pairs(options_json, pair_block.copy())

    pairs = system.pairs(options)
    assert len(pairs.samples) == 1
    assert len(pairs.properties) == 1

    known = system.known_pairs()
    assert len(known) == 2
    assert known[0].cutoff == 1.0
    assert known[0].full_list is True
    assert known[0].strict is False
    assert known[0].requestors == ["test"]


def test_system_missing_pairs_is_an_error():
    types = np.array([1, 4], dtype=np.int32)
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    cell = np.zeros((3, 3), dtype=np.float64)
    pbc = np.array([False, False, False])
    system = System("nm", types, positions, cell, pbc)
    missing = PairListOptions(cutoff=9.0, full_list=False)
    with pytest.raises(MetatomicError):
        system.pairs(missing)


def test_system_custom_data(system, custom_data):
    system.add_custom_data("test::my_data", custom_data.copy())

    data = system.custom_data("test::my_data")
    assert len(data.keys) == 1

    with pytest.raises(MetatomicError):
        system.custom_data("test::no_such_data")

    system.add_custom_data("test::other_data", custom_data.copy())
    names = sorted(system.known_custom_data())
    assert names == ["test::my_data", "test::other_data"]


def test_system_additions_require_released_views(system, pair_block, custom_data):
    options = PairListOptions(cutoff=1.0, full_list=True)
    positions = system.positions
    with pytest.raises(MetatomicError, match="outstanding borrowed views"):
        system.add_pairs(options, pair_block.copy())
    with pytest.raises(MetatomicError, match="outstanding borrowed views"):
        system.add_custom_data("test::my_data", custom_data.copy())
    assert system.size == 4

    del positions
    system.add_pairs(options, pair_block.copy())
    system.add_custom_data("test::my_data", custom_data.copy())
    assert len(system.known_pairs()) == 1
    assert system.known_custom_data() == ["test::my_data"]


def test_system_failed_additions_keep_ownership(system, pair_block, custom_data):
    options = PairListOptions(cutoff=1.0, full_list=True)
    system.add_pairs(options, pair_block.copy())
    system.add_custom_data("test::my_data", custom_data.copy())

    with pytest.raises(MetatomicError):
        system.add_pairs(options, pair_block.copy())
    with pytest.raises(MetatomicError):
        system.add_custom_data("test::my_data", custom_data.copy())

    assert system.size == 4
    other = PairListOptions(cutoff=2.0, full_list=False)
    system.add_pairs(other, pair_block.copy())
    system.add_custom_data("test::other_data", custom_data.copy())
    assert len(system.known_pairs()) == 2
    assert sorted(system.known_custom_data()) == ["test::my_data", "test::other_data"]


def test_system_ownership(system):
    raw = system.as_mta_system_t()

    view = System.unsafe_view_from_ptr(raw)
    assert view.size == 4
    assert view.arrays_backend is None

    with pytest.raises(ValueError, match="view of a system owned elsewhere"):
        view.release()

    del view
    assert system.size == 4

    raw = system.release()
    owned = System.unsafe_from_ptr(raw)
    assert owned.size == 4
    assert owned.arrays_backend is None

    with pytest.raises(ValueError, match="released"):
        system.size
    assert repr(system) == "System(<released>)"


def test_system_arrays_backend_requires_initialization(system, pair_block):
    raw = system.as_mta_system_t()
    view = System.unsafe_view_from_ptr(raw)

    message = (
        "Arrays backend not initialized, please set it with System.set_arrays_backend"
    )
    with pytest.raises(ValueError, match=message):
        view.positions

    # A failed getter must not leave a C-level borrow that blocks additions.
    system.add_pairs(PairListOptions(cutoff=1.0, full_list=True), pair_block.copy())
    assert len(system.known_pairs()) == 1

    view.set_arrays_backend("numpy")
    assert view.arrays_backend == "numpy"
    assert isinstance(view.positions, np.ndarray)
    assert view.positions[3, 0] == 10.0


def test_system_set_arrays_backend_unknown(system):
    with pytest.raises(ValueError, match="Unknown arrays backend: nope"):
        system.set_arrays_backend("nope")


def test_system_arrays_backend_dlpack(system):
    from ctypes_dlpack import DLPackArray

    system.set_arrays_backend("dlpack")
    assert system.arrays_backend == "dlpack"
    assert isinstance(system.positions, DLPackArray)


def test_system_arrays_backend_torch():
    torch = pytest.importorskip("torch")

    types = torch.tensor([1, 4, 7, 10], dtype=torch.int32)
    positions = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=torch.float64,
    )
    cell = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=torch.float64,
    )
    pbc = torch.tensor([True, False, True])

    system = System("nm", types, positions, cell, pbc)
    assert system.arrays_backend == "torch"
    assert isinstance(system.types, torch.Tensor)
    assert isinstance(system.positions, torch.Tensor)
    assert isinstance(system.cell, torch.Tensor)
    assert isinstance(system.pbc, torch.Tensor)
    assert system.positions[3, 0] == 10.0


def test_system_set_arrays_backend_torch(system):
    torch = pytest.importorskip("torch")

    system.set_arrays_backend("torch")
    assert system.arrays_backend == "torch"
    assert isinstance(system.positions, torch.Tensor)
    assert system.positions[3, 0] == 10.0


def test_system_torch_compile_squared_sum_positions(system):
    # Constructing a System or reading its getters inside torch.compile is
    # not supported (the getters go through ctypes). Arrays taken out of a
    # System can still be used in a compiled function.
    torch = pytest.importorskip("torch")
    system.set_arrays_backend("torch")

    def squared_sum_positions(positions):
        return torch.sum(positions) ** 2

    positions = system.positions
    expected = squared_sum_positions(positions)
    compiled = torch.compile(squared_sum_positions, backend="eager")
    torch.testing.assert_close(compiled(positions), expected)


def test_system_arrays_backend_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy

    types = jnp.array([1, 4, 7, 10], dtype=jnp.int32)
    positions = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=jnp.float64,
    )
    cell = jnp.array(
        [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=jnp.float64,
    )
    pbc = jnp.array([True, False, True])

    system = System("nm", types, positions, cell, pbc)
    assert system.arrays_backend == "jax"
    assert isinstance(system.positions, jax.Array)
    assert float(system.positions[3, 0]) == 10.0


def test_system_set_arrays_backend_jax(system):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)

    system.set_arrays_backend("jax")
    assert system.arrays_backend == "jax"
    assert isinstance(system.positions, jax.Array)
    assert float(system.positions[3, 0]) == 10.0
