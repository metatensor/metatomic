import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock

from metatomic.torch import NeighborListOptions, System, register_autograd_neighbors


try:
    import ase
    import ase.neighborlist

    HAVE_ASE = True
except ImportError:
    HAVE_ASE = False


ALL_DEVICES = ["cpu"]
if torch.cuda.is_available():
    ALL_DEVICES.append("cuda")


def test_neighbor_list_options():
    options = NeighborListOptions(3.4, True, True, "hello")

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

    assert NeighborListOptions(3.4, True, True, "a") == NeighborListOptions(
        3.4, True, True, "b"
    )
    assert NeighborListOptions(3.4, True, True) != NeighborListOptions(3.4, False, True)
    assert NeighborListOptions(3.4, False, True) != NeighborListOptions(
        3.4, False, False
    )
    assert NeighborListOptions(3.4, True, True) != NeighborListOptions(3.5, True, True)

    expected = "NeighborListOptions(cutoff=3.400000, full_list=True, strict=True)"
    assert str(options) == expected

    expected = """NeighborListOptions
    cutoff: 3.400000
    full_list: True
    strict: True
    requested by:
        - hello
        - another one
"""

    assert repr(options) == expected


def compute_neighbors_with_ase(system, options):
    # options.strict is ignored by this function, since `ase.neighborlist.neighbor_list`
    # only computes strict NL, and these are valid even with `strict=False`

    dtype = system.positions.dtype
    device = system.positions.device

    atoms = ase.Atoms(
        numbers=system.types.cpu().numpy(),
        positions=system.positions.cpu().detach().numpy(),
        cell=system.cell.cpu().detach().numpy(),
        pbc=system.pbc.cpu().numpy(),
    )

    nl_i, nl_j, nl_S, nl_D = ase.neighborlist.neighbor_list(
        "ijSD",
        atoms,
        cutoff=options.engine_cutoff(engine_length_unit="angstrom"),
    )

    if not options.full_list:
        # The pair selection code here below avoids a relatively slow loop over
        # all pairs to improve performance
        reject_condition = (
            # we want a half neighbor list, so drop all duplicated neighbors
            (nl_j < nl_i)
            | (
                (nl_i == nl_j)
                & (
                    # only create pairs with the same atom twice if the pair spans more
                    # than one unit cell
                    ((nl_S[:, 0] == 0) & (nl_S[:, 1] == 0) & (nl_S[:, 2] == 0))
                    # When creating pairs between an atom and one of its periodic
                    # images, the code generates multiple redundant pairs
                    # (e.g. with shifts 0 1 1 and 0 -1 -1); and we want to only keep one
                    # of these. We keep the pair in the positive half plane of shifts.
                    | (
                        (nl_S.sum(axis=1) < 0)
                        | (
                            (nl_S.sum(axis=1) == 0)
                            & (
                                (nl_S[:, 2] < 0)
                                | ((nl_S[:, 2] == 0) & (nl_S[:, 1] < 0))
                            )
                        )
                    )
                )
            )
        )
        selected = np.logical_not(reject_condition)
        nl_i = nl_i[selected]
        nl_j = nl_j[selected]
        nl_S = nl_S[selected]
        nl_D = nl_D[selected]

    samples = np.concatenate([nl_i[:, None], nl_j[:, None], nl_S], axis=1)

    distances = torch.from_numpy(nl_D).to(dtype=dtype, device=device)

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
            values=torch.from_numpy(samples).to(dtype=torch.int32, device=device),
            assume_unique=True,
        ),
        components=[Labels.range("xyz", 3).to(device)],
        properties=Labels.range("distance", 1).to(device),
    )


@pytest.mark.skipif(not HAVE_ASE, reason="this tests requires ASE neighbor list")
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
        neighbors = compute_neighbors_with_ase(system, options)
        register_autograd_neighbors(system, neighbors, check_consistency=True)
        return neighbors.values.sum()

    options = NeighborListOptions(cutoff=2.0, full_list=False, strict=True)
    torch.autograd.gradcheck(
        compute,
        (positions, cell, options),
        fast_mode=True,
    )

    options = NeighborListOptions(cutoff=2.0, full_list=True, strict=True)
    torch.autograd.gradcheck(
        compute,
        (positions, cell, options),
        fast_mode=True,
    )


@pytest.mark.skipif(not HAVE_ASE, reason="this tests requires ASE neighbor list")
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
    options = NeighborListOptions(cutoff=2.0, full_list=False, strict=True)
    neighbors = compute_neighbors_with_ase(system, options)
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
    neighbors = compute_neighbors_with_ase(system, options)
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
