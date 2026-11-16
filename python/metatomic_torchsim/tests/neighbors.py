import ase.build
import ase.neighborlist
import numpy as np
import pytest
import torch
from metatensor.torch import Labels

from metatomic.torch import NeighborListOptions, System
from metatomic_torchsim._neighbors import HAS_NVALCHEMIOPS, AllNeighborsCalculator


@pytest.mark.skipif(
    not (HAS_NVALCHEMIOPS and torch.cuda.is_available()),
    reason="nvalchemiops neighbor list is only used on CUDA",
)
def test_mixed_pbc_on_cuda():
    atoms = ase.build.fcc111("Ni", size=(2, 2, 3), vacuum=10.0)
    atoms.set_pbc((True, True, False))
    cutoff = 4.0

    # metatomic requires a zero cell vector along non-periodic directions
    cell = atoms.cell.array.copy()
    cell[2] = 0.0

    system = System(
        types=torch.tensor(atoms.numbers, dtype=torch.int32, device="cuda"),
        positions=torch.tensor(atoms.positions, dtype=torch.float64, device="cuda"),
        cell=torch.tensor(cell, dtype=torch.float64, device="cuda"),
        pbc=torch.tensor(atoms.pbc, device="cuda"),
    )
    options = NeighborListOptions(cutoff=cutoff, full_list=True, strict=True)

    AllNeighborsCalculator([options]).compute([system])
    neighbors = system.get_neighbor_list(options)

    first, second, shifts, vectors = ase.neighborlist.neighbor_list(
        "ijSD", atoms, cutoff
    )
    expected = Labels(
        neighbors.samples.names,
        torch.from_numpy(np.column_stack([first, second, shifts])).to(torch.int32),
    ).to(device="cuda")

    assert len(neighbors.samples) == len(expected)

    for sample_i, sample in enumerate(expected):
        position = neighbors.samples.position(sample)
        assert position is not None

        actual = neighbors.values[position].reshape(3).cpu()
        assert torch.allclose(actual, torch.from_numpy(vectors[sample_i]))
