import metatensor.torch
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import PairListOptions, System

from . import _tests_utils


@pytest.fixture
def types():
    return torch.tensor([1, -2, 3, 1, 1, 1, 3, 3])


@pytest.fixture
def positions():
    return torch.rand((8, 3))


@pytest.fixture
def cell():
    return torch.tensor([[12.0, 0, 0], [0, 12.3, 0], [0, 0, 10]])


@pytest.fixture
def pbc():
    return torch.tensor([True, True, True])


@pytest.fixture
def system(types, positions, cell, pbc):
    return System(types=types, positions=positions, cell=cell, pbc=pbc)


@pytest.fixture
def neighbors():
    return TensorBlock(
        values=torch.zeros(2, 3, 1),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            torch.tensor(
                [
                    (0, 1, 0, 0, 0),
                    (0, 2, 1, 0, -1),
                ]
            ),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )


def test_system(types, positions, cell, pbc, neighbors):
    system = System(types, positions, cell, pbc)

    assert torch.all(system.types == types)
    assert torch.all(system.positions == positions)
    assert torch.all(system.cell == cell)
    assert torch.all(system.pbc == pbc)

    expected = "System with 8 atoms, periodic cell: [12, 0, 0, 0, 12.3, 0, 0, 0, 10]"
    assert str(system) == expected
    assert repr(system) == expected

    system = System(
        types,
        positions,
        cell=torch.zeros_like(cell),
        pbc=torch.tensor([False, False, False]),
    )
    expected = "System with 8 atoms, non periodic"
    assert str(system) == expected
    assert repr(system) == expected

    options = PairListOptions(cutoff=3.5, full_list=False, strict=True)
    system.add_pairs(options, neighbors)

    assert metatensor.torch.equal_block(system.pairs(options), neighbors)

    message = "no pair list found for the given options"
    with pytest.raises(RuntimeError, match=message):
        system.pairs(PairListOptions(cutoff=3.5, full_list=True, strict=True))

    message = "the pair list for these options already exists in this system"
    with pytest.raises(RuntimeError, match=message):
        system.add_pairs(options, neighbors)

    assert system.known_pairs() == [
        PairListOptions(cutoff=3.5, full_list=False, strict=True)
    ]


def test_deprecated_system_functions(system, neighbors):
    options = PairListOptions(cutoff=3.5, full_list=False, strict=True)
    tensor = TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.zeros(3, 10),
                samples=Labels.range("foo", 3),
                components=[],
                properties=Labels.range("distance", 10),
            )
        ],
    )

    deprecated_calls = [
        (
            "add_neighbor_list",
            (options, neighbors),
            r"`add_neighbor_list` is deprecated, use `add_pairs` instead",
        ),
        (
            "get_neighbor_list",
            (options,),
            r"`get_neighbor_list` is deprecated, use `pairs` instead",
        ),
        (
            "known_neighbor_lists",
            (),
            r"`known_neighbor_lists` is deprecated, use `known_pairs` instead",
        ),
        (
            "add_data",
            ("custom::data_name", tensor),
            r"`add_data` is deprecated, use `add_custom_data` instead",
        ),
        (
            "get_data",
            ("custom::data_name",),
            r"`get_data` is deprecated, use `custom_data` instead",
        ),
        (
            "known_data",
            (),
            r"`known_data` is deprecated, use `known_custom_data` instead",
        ),
    ]

    # make sure the warning is emitted even for the second call to a deprecated
    # function, since these functions are otherwise only deprecated once per process
    torch.set_warn_always(True)
    try:
        for name, args, message in deprecated_calls:
            with pytest.warns(DeprecationWarning, match=message):
                getattr(system, name)(*args)
    finally:
        torch.set_warn_always(False)

    assert metatensor.torch.equal_block(system.pairs(options), neighbors)
    assert system.known_pairs() == [options]
    assert system.known_custom_data() == ["custom::data_name"]


def test_custom_data(system):
    block = TensorBlock(
        values=torch.zeros(3, 10),
        samples=Labels.range("foo", 3),
        components=[],
        properties=Labels.range("distance", 10),
    )
    tensor = TensorMap(Labels.single(), [block])

    system.add_custom_data("custom::data_name", tensor)
    assert system.known_custom_data() == ["custom::data_name"]

    message = "custom data can not be named 'positions'"
    with pytest.raises(RuntimeError, match=message):
        system.add_custom_data("positions", tensor)

    message = "custom data can not be named 'position'"
    with pytest.raises(RuntimeError, match=message):
        system.add_custom_data("position", tensor)

    message = "custom data 'custom::data_name' is already present in this system"
    with pytest.raises(RuntimeError, match=message):
        system.add_custom_data("custom::data_name", tensor)

    assert metatensor.torch.equal(system.custom_data("custom::data_name"), tensor)


def test_data_validation(types, positions, cell, pbc):
    # this should run without error:
    system = System(types, positions, cell, pbc)

    # ===== device checks ===== #
    message = "`types`, `positions`, `cell`, and `pbc` must be on the same device"
    with pytest.raises(RuntimeError, match=message):
        System(types.to(device="meta"), positions, cell, pbc)

    with pytest.raises(RuntimeError, match=message):
        system.types = types.to(device="meta")

    with pytest.raises(RuntimeError, match=message):
        System(types, positions.to(device="meta"), cell, pbc)

    with pytest.raises(RuntimeError, match=message):
        system.positions = positions.to(device="meta")

    with pytest.raises(RuntimeError, match=message):
        System(types, positions, cell.to(device="meta"), pbc)

    with pytest.raises(RuntimeError, match=message):
        system.cell = cell.to(device="meta")

    with pytest.raises(RuntimeError, match=message):
        System(types, positions, cell, pbc.to(device="meta"))

    with pytest.raises(RuntimeError, match=message):
        system.pbc = pbc.to(device="meta")

    # ===== types checks ===== #
    message = (
        "`types` must be a \\(n_atoms,\\) tensor, got a tensor with shape \\[8, 1\\]"
    )
    with pytest.raises(RuntimeError, match=message):
        System(types.reshape(-1, 1), positions, cell, pbc)

    with pytest.raises(RuntimeError, match=message):
        system.types = types.reshape(-1, 1)

    message = "`types` must be a tensor of integers, got torch.float64 instead"
    with pytest.raises(ValueError, match=message):
        System(types.to(dtype=torch.float64), positions, cell, pbc)

    with pytest.raises(ValueError, match=message):
        system.types = types.to(dtype=torch.float64)

    # ===== positions checks ===== #
    message = (
        "`positions` must be a \\(n_atoms x 3\\) tensor, "
        "got a tensor with shape \\[1, 8, 3\\]"
    )
    with pytest.raises(RuntimeError, match=message):
        System(types, positions.reshape(1, -1, 3), cell, pbc)

    with pytest.raises(RuntimeError, match=message):
        system.positions = positions.reshape(1, -1, 3)

    message = (
        "`positions` must be a \\(n_atoms x 3\\) tensor, "
        "got a tensor with shape \\[8, 3\\]"
    )
    with pytest.raises(RuntimeError, match=message):
        System(torch.hstack([types, types]), positions, cell, pbc)

    message = (
        "`positions` must be a \\(n_atoms x 3\\) tensor, "
        "got a tensor with shape \\[16, 3\\]"
    )
    with pytest.raises(RuntimeError, match=message):
        system.positions = torch.vstack([positions, positions])

    message = "`positions` must be a tensor of 32 or 64-bit floating point data"
    with pytest.raises(RuntimeError, match=message):
        System(types, positions.to(dtype=torch.int32), cell, pbc)

    message = "`cell` must have the same dtype as `positions`, got f32 and f64"
    with pytest.raises(RuntimeError, match=message):
        system.positions = positions.to(dtype=torch.float64)

    # ===== cell checks ===== #
    message = (
        "`cell` must be a \\(3 x 3\\) tensor, got a tensor with shape \\[3, 1, 3\\]"
    )
    with pytest.raises(RuntimeError, match=message):
        System(types, positions, cell.reshape(3, 1, 3), pbc)

    with pytest.raises(RuntimeError, match=message):
        system.cell = cell.reshape(3, 1, 3)

    message = "`cell` must be a \\(3 x 3\\) tensor, got a tensor with shape \\[6, 3\\]"
    with pytest.raises(RuntimeError, match=message):
        System(types, positions, torch.vstack([cell, cell]), pbc)

    with pytest.raises(RuntimeError, match=message):
        system.cell = torch.vstack([cell, cell])

    message = "`cell` must have the same dtype as `positions`, got i32 and f32"
    with pytest.raises(RuntimeError, match=message):
        System(types, positions, cell.to(dtype=torch.int32), pbc)

    message = "`cell` must have the same dtype as `positions`, got f64 and f32"
    with pytest.raises(RuntimeError, match=message):
        system.cell = cell.to(dtype=torch.float64)

    # ===== pbc checks ===== #
    message = "`pbc` must contain 3 entries, got a tensor with shape \\[3, 1\\]"
    with pytest.raises(RuntimeError, match=message):
        System(types, positions, cell, pbc.reshape(-1, 1))

    with pytest.raises(RuntimeError, match=message):
        system.pbc = pbc.reshape(-1, 1)

    message = "`pbc` must be a tensor of booleans"
    with pytest.raises(RuntimeError, match=message):
        System(types, positions, cell, pbc.to(dtype=torch.int32))

    with pytest.raises(RuntimeError, match=message):
        system.pbc = pbc.to(dtype=torch.int32)


def test_neighbors_validation(system):
    options = PairListOptions(cutoff=3.5, full_list=False, strict=True)

    message = (
        "invalid samples for `pairs`: the samples names must be 'first_atom', "
        "'second_atom', 'cell_shift_a', 'cell_shift_b', 'cell_shift_c'"
    )
    with pytest.raises(RuntimeError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 1),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                ],
                torch.tensor([(0, 1, 0, 0)]),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )

        system.add_pairs(options, neighbors)

    message = (
        "invalid components for `pairs`: "
        "there should be a single 'xyz'=\\[0, 1, 2\\] component"
    )
    with pytest.raises(RuntimeError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 1),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor([(0, 1, 0, 0, 0)]),
            ),
            components=[Labels.range("a", 3)],
            properties=Labels.range("distance", 1),
        )

        system.add_pairs(options, neighbors)

    message = (
        "invalid properties for `pairs`: there should be a single 'distance'=0 property"
    )
    with pytest.raises(RuntimeError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 2),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor([(0, 1, 0, 0, 0)]),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 2),
        )

        system.add_pairs(options, neighbors)

    message = "`pairs` should not have any gradients"
    with pytest.raises(RuntimeError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 1),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor([(0, 1, 0, 0, 0)]),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )

        neighbors.add_gradient(
            "gradient",
            TensorBlock(
                values=torch.zeros(1, 3, 1),
                samples=Labels.range("sample", 1),
                components=[Labels.range("xyz", 3)],
                properties=Labels.range("distance", 1),
            ),
        )

        system.add_pairs(options, neighbors)

    message = (
        "`pairs` device \\(ExtDev:0\\) does not match this system's device \\(CPU:0\\)"
    )
    with pytest.raises(RuntimeError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 1).to(device="meta"),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor([(0, 1, 0, 0, 0)]),
            ).to(device="meta"),
            components=[Labels.range("xyz", 3).to(device="meta")],
            properties=Labels.range("distance", 1).to(device="meta"),
        )

        system.add_pairs(options, neighbors)

    message = "`pairs` dtype \\(f64\\) does not match this system's dtype \\(f32\\)"
    with pytest.raises(RuntimeError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 1).to(dtype=torch.float64),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor([(0, 1, 0, 0, 0)]),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )

        system.add_pairs(options, neighbors)


def test_to(system, neighbors):
    options = PairListOptions(cutoff=3.5, full_list=False, strict=True)
    system.add_pairs(options, neighbors)

    system.add_custom_data(
        "custom::test_data", TensorMap(Labels.single(), [neighbors.copy()])
    )

    assert system.device.type == torch.device("cpu").type
    check_dtype(system, torch.float32)

    # we use non_blocking=True for some of the calls to `.to` below as a smoke test,
    # making sure the parameter is accepted by this function
    converted = system.to(dtype=torch.float64, non_blocking=True)
    check_dtype(converted, torch.float64)

    devices = ["meta", torch.device("meta")]
    if _tests_utils.can_use_mps_backend():
        devices.append("mps")
        devices.append(torch.device("mps"))

    if torch.cuda.is_available():
        devices.append("cuda")
        devices.append("cuda:0")
        devices.append(torch.device("cuda"))

    for device in devices:
        moved = system.to(device=device)
        assert moved.device.type == torch.device(device).type

    # check that the code handles both positional and keyword arguments
    device = "meta"
    moved = system.to(device, dtype=torch.float32)
    moved = system.to(torch.float32, device, non_blocking=True)
    moved = system.to(torch.float32, device=device)
    moved = system.to(device, torch.float32)

    message = "can not give a device twice in `System.to`"
    with pytest.raises(ValueError, match=message):
        moved = system.to("meta", device="meta")

    message = "can not give a dtype twice in `System.to`"
    with pytest.raises(ValueError, match=message):
        moved = system.to(torch.float32, dtype=torch.float32)

    message = "unexpected type in `System.to`: Tensor"
    with pytest.raises(TypeError, match=message):
        moved = system.to(torch.tensor([0]))


# This function only works in script mode, because `block.dtype` is always an `int`, and
# `torch.dtype` is only an int in script mode.
@torch.jit.script
def check_dtype(system: System, dtype: torch.dtype):
    assert system.dtype == dtype


def test_partial_pbc():
    system = System(
        torch.tensor([1]),
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        pbc=torch.tensor([True, False, True]),
    )

    message = (
        "invalid cell: for non-periodic dimensions, "
        "the corresponding cell vector must be zero"
    )
    with pytest.raises(RuntimeError, match=message):
        System(
            torch.tensor([1]),
            torch.tensor([[1.0, 1.0, 1.0]]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            pbc=torch.tensor([True, False, True]),
        )

    with pytest.raises(RuntimeError, match=message):
        system.pbc = torch.tensor([True, True, False])
