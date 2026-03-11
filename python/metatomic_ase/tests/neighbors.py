import glob
import json
import os

import pytest
import torch
from metatensor.torch import Labels, TensorBlock

from metatomic.torch import (
    NeighborListOptions,
    System,
)
from metatomic_ase._neighbors import (
    _compute_requested_neighbors_nvalchemi,
    _compute_requested_neighbors_vesin,
)

from ._tests_utils import ALL_DEVICE_DTYPE


def _read_neighbor_check(path):
    with open(path) as fd:
        data = json.load(fd)

    dtype = torch.float64

    positions = torch.tensor(data["system"]["positions"], dtype=dtype).reshape(-1, 3)
    system = System(
        types=torch.tensor([1] * positions.shape[0], dtype=torch.int32),
        positions=positions,
        cell=torch.tensor(data["system"]["cell"], dtype=dtype),
        pbc=torch.tensor([True, True, True]),
    )

    options = NeighborListOptions(
        cutoff=data["options"]["cutoff"],
        full_list=data["options"]["full_list"],
        # ASE can only compute strict NL
        strict=True,
    )

    samples = torch.tensor(
        data["expected-neighbors"]["samples"], dtype=torch.int32
    ).reshape(-1, 5)
    distances = torch.tensor(
        data["expected-neighbors"]["distances"], dtype=dtype
    ).reshape(-1, 3, 1)

    neighbors = TensorBlock(
        values=distances,
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            samples,
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )

    return system, options, neighbors


def _check_same_set_of_neighbors(expected, actual, full_list):
    assert expected.samples.names == actual.samples.names
    assert len(expected.samples) == len(actual.samples)

    for sample_i, sample in enumerate(expected.samples):
        sign = 1.0
        position = actual.samples.position(sample)

        if position is None and not full_list:
            # try looking for the inverse pair
            sign = -1.0
            position = actual.samples.position(
                [sample[1], sample[0], -sample[2], -sample[3], -sample[4]]
            )

        if position is None:
            raise AssertionError(f"missing expected neighbors sample: {sample}")

        assert torch.allclose(expected.values[sample_i], sign * actual.values[position])


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
@pytest.mark.parametrize(
    "neighbor_fn",
    [_compute_requested_neighbors_vesin, _compute_requested_neighbors_nvalchemi],
)
def test_neighbor_list_adapter(device, dtype, neighbor_fn):
    if neighbor_fn == _compute_requested_neighbors_nvalchemi and device != "cuda":
        pytest.skip("nvalchemiops neighbor list is only implemented for CUDA")

    HERE = os.path.realpath(os.path.dirname(__file__))
    test_files = os.path.join(
        HERE, "..", "..", "..", "..", "metatensor-torch", "tests", "neighbor-checks"
    )

    for path in glob.glob(os.path.join(test_files, "*.json")):
        system, options, expected_neighbors = _read_neighbor_check(path)
        system = system.to(device=device, dtype=dtype)
        expected_neighbors = expected_neighbors.to(device=device, dtype=dtype)

        neighbor_fn([system], [options], check_consistency=True)
        _check_same_set_of_neighbors(
            expected_neighbors, system.get_neighbor_list(options), options.full_list
        )
