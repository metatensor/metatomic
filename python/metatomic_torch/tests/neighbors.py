import pytest
import torch
import vesin
from metatensor.torch import Labels, TensorBlock

from metatomic.torch import PairListOptions, System, register_autograd_neighbors


ALL_DEVICES = ["cpu"]
if torch.cuda.is_available():
    ALL_DEVICES.append("cuda")


def test_pair_list_options():
    options = PairListOptions(3.4, True, True, "hello")

    assert options.cutoff == 3.4
    assert options.full_list
    assert options.strict
    assert options.requestors() == ["hello"]

    options.add_requestor("another one")
    assert options.requestors() == ["hello", "another one"]

    # No empty requestors, no duplicated requestors
    options.add_requestor("")
    options.add_requestor("hello")
    assert options.requestors() == ["hello", "another one"]

    assert PairListOptions(3.4, True, True, "a") == PairListOptions(
        3.4, True, True, "b"
    )
    assert PairListOptions(3.4, True, True) != PairListOptions(3.4, False, True)
    assert PairListOptions(3.4, False, True) != PairListOptions(3.4, False, False)
    assert PairListOptions(3.4, True, True) != PairListOptions(3.5, True, True)

    expected = "PairListOptions(cutoff=3.400000, full_list=True, strict=True)"
    assert str(options) == expected

    expected = """PairListOptions
    cutoff: 3.400000
    full_list: True
    strict: True
    requested by:
        - hello
        - another one
"""

    assert repr(options) == expected


def compute_neighbors_with_vesin(system, options):
    # options.strict is ignored by this function, since vesin only computes strict NL,
    # and these are valid even with `strict=False`

    device = system.positions.device

    # vesin runs on the same device as the input tensors (including CUDA), and
    # returns torch tensors on this device
    calculator = vesin.NeighborList(
        cutoff=options.engine_cutoff(engine_length_unit="angstrom"),
        full_list=options.full_list,
    )
    nl_P, nl_S, nl_D = calculator.compute(
        points=system.positions.detach(),
        box=system.cell.detach(),
        periodic=system.pbc,
        quantities="PSD",
    )

    return TensorBlock(
        values=nl_D.reshape(-1, 3, 1),
        samples=Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            # pairs are returned as uint64, which can not be promoted with int32
            values=torch.hstack([nl_P.to(torch.int32), nl_S]),
            assume_unique=True,
        ),
        components=[Labels.range("xyz", 3).to(device)],
        properties=Labels.range("distance", 1).to(device),
    )


def synthetic_neighbors(system):
    # A few pairs with distance vectors matching the system positions and cell. This
    # is not a complete neighbor list, but it is enough for the consistency checks in
    # `register_autograd_neighbors`.
    device = system.positions.device

    pairs = torch.tensor(
        [[0, 1, 0, 0, 0], [0, 2, 1, 0, 0], [1, 2, 0, -1, 1], [2, 2, 0, 0, 1]],
        device=device,
    )
    positions = system.positions.detach()
    shifts = pairs[:, 2:].to(positions.dtype) @ system.cell.detach()
    distances = positions[pairs[:, 1]] - positions[pairs[:, 0]] + shifts

    return TensorBlock(
        values=distances.reshape(-1, 3, 1),
        samples=Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=pairs.to(torch.int32),
        ),
        components=[Labels.range("xyz", 3).to(device)],
        properties=Labels.range("distance", 1).to(device),
    )


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_neighbors_autograd(device):
    torch.manual_seed(0xDEADBEEF)
    n_atoms = 20
    approx_cell_size = 6.0
    positions = approx_cell_size * torch.rand(
        n_atoms, 3, dtype=torch.float64, requires_grad=True, device=device
    )
    cell = approx_cell_size * (
        torch.eye(3, dtype=torch.float64, device=device)
        + 0.1 * torch.rand(3, 3, dtype=torch.float64, device=device)
    )
    cell.requires_grad = True

    def compute(positions, cell, options):
        system = System(
            torch.tensor([6] * len(positions), device=positions.device),
            positions,
            cell,
            pbc=torch.tensor([True, True, True], device=positions.device),
        )
        neighbors = compute_neighbors_with_vesin(system, options)
        register_autograd_neighbors(system, neighbors, check_consistency=True)
        return neighbors.values.sum()

    options = PairListOptions(cutoff=2.0, full_list=False, strict=True)
    torch.autograd.gradcheck(
        compute,
        (positions, cell, options),
        fast_mode=True,
    )

    options = PairListOptions(cutoff=2.0, full_list=True, strict=True)
    torch.autograd.gradcheck(
        compute,
        (positions, cell, options),
        fast_mode=True,
    )


def test_neighbor_autograd_errors():
    n_atoms = 20
    cell_size = 6.0
    positions = cell_size * torch.rand(
        n_atoms, 3, dtype=torch.float64, requires_grad=True
    )
    cell = cell_size * torch.eye(3, dtype=torch.float64, requires_grad=True)

    system = System(
        torch.tensor([6] * len(positions)),
        positions,
        cell,
        pbc=torch.tensor([True, True, True]),
    )
    neighbors = synthetic_neighbors(system)
    register_autograd_neighbors(system, neighbors, check_consistency=True)

    system = System(
        torch.tensor([6] * len(positions)),
        positions,
        cell,
        pbc=torch.tensor([True, True, True]),
    )
    message = (
        "`neighbors` is already part of a computational graph, "
        "detach it before calling `register_autograd_neighbors\\(\\)`"
    )
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors)

    message = (
        r"one neighbor pair does not match its metadata: the pair between atom \d+ and "
        r"atom \d+ for the \[.*?\] cell shift should have a distance vector "
        r"of \[.*?\] but has a distance vector of \[.*?\]"
    )

    system = System(
        torch.tensor([6] * len(positions)),
        positions,
        cell,
        pbc=torch.tensor([True, True, True]),
    )
    neighbors = synthetic_neighbors(system)
    neighbors.values[:] *= 3
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors, check_consistency=True)
    neighbors.values[:] /= 3

    system = System(
        torch.tensor([6] * len(positions)),
        positions,
        cell,
        pbc=torch.tensor([True, True, True]),
    )
    neighbors = neighbors.to(torch.float32)
    message = (
        "`system` and `neighbors` must have the same dtype, "
        "got torch.float64 and torch.float32"
    )
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors, check_consistency=True)

    message = (
        r"`system` and `neighbors` must be on the same device, got meta and [\w:]+"
    )
    system = System(
        torch.tensor([6] * len(positions), device="meta"),
        positions.to(torch.device("meta")),
        cell.to(torch.device("meta")),
        pbc=torch.tensor([True, True, True]).to(torch.device("meta")),
    )
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors, check_consistency=True)


def test_torch_script():
    # make sure functions that have side effects are properly included in the
    # TorchScript code

    @torch.jit.script
    def test_function(system: System, neighbors: TensorBlock, check_consistency: bool):
        register_autograd_neighbors(system, neighbors, check_consistency)

    assert "ops.metatomic.register_autograd_neighbors" in test_function.code
