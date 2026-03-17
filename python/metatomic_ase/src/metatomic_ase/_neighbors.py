from typing import List

import torch
import vesin.metatomic
from metatensor.torch import Labels, TensorBlock

from metatomic.torch import NeighborListOptions, System


try:
    from nvalchemiops.torch.neighbors import neighbor_list as nvalchemi_neighbor_list

    HAS_NVALCHEMIOPS = True
except ImportError:
    HAS_NVALCHEMIOPS = False


def _compute_requested_neighbors(
    systems: List[System],
    requested_options: List[NeighborListOptions],
    check_consistency=False,
) -> List[System]:
    """
    Compute all neighbor lists requested by ``model`` and store them inside the systems.
    """
    can_use_nvalchemi = HAS_NVALCHEMIOPS and all(
        system.device.type == "cuda" for system in systems
    )

    if can_use_nvalchemi:
        full_nl_options = []
        half_nl_options = []
        for options in requested_options:
            if options.full_list:
                full_nl_options.append(options)
            else:
                half_nl_options.append(options)

        # Do the full neighbor lists with nvalchemi, and the rest with vesin
        systems = _compute_requested_neighbors_nvalchemi(
            systems=systems,
            requested_options=full_nl_options,
        )
        systems = _compute_requested_neighbors_vesin(
            systems=systems,
            requested_options=half_nl_options,
            check_consistency=check_consistency,
        )
    else:
        systems = _compute_requested_neighbors_vesin(
            systems=systems,
            requested_options=requested_options,
            check_consistency=check_consistency,
        )

    return systems


def _compute_requested_neighbors_vesin(
    systems: List[System],
    requested_options: List[NeighborListOptions],
    check_consistency=False,
) -> List[System]:
    """
    Compute all neighbor lists requested by ``model`` and store them inside the systems,
    using vesin.
    """

    system_devices = []
    moved_systems = []
    for system in systems:
        system_devices.append(system.device)
        if system.device.type not in ["cpu", "cuda"]:
            moved_systems.append(system.to(device="cpu"))
        else:
            moved_systems.append(system)

    vesin.metatomic.compute_requested_neighbors_from_options(
        systems=moved_systems,
        system_length_unit="angstrom",
        options=requested_options,
        check_consistency=check_consistency,
    )

    systems = []
    for system, device in zip(moved_systems, system_devices, strict=True):
        systems.append(system.to(device=device))

    return systems


def _compute_requested_neighbors_nvalchemi(systems, requested_options):
    """
    Compute all neighbor lists requested by ``model`` and store them inside the systems,
    using nvalchemiops. This function should only be called if all systems are on CUDA
    and all neighbor list options require a full neighbor list.
    """

    for options in requested_options:
        assert options.full_list
        for system in systems:
            assert system.device.type == "cuda"

            edge_index, _, S = nvalchemi_neighbor_list(
                system.positions,
                options.engine_cutoff("angstrom"),
                cell=system.cell,
                pbc=system.pbc,
                return_neighbor_list=True,
            )
            D = (
                system.positions[edge_index[1]]
                - system.positions[edge_index[0]]
                + S.to(system.cell.dtype) @ system.cell
            )
            P = edge_index.T

            neighbors = TensorBlock(
                D.reshape(-1, 3, 1),
                samples=Labels(
                    names=[
                        "first_atom",
                        "second_atom",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    values=torch.hstack([P, S]),
                ),
                components=[
                    Labels("xyz", torch.tensor([[0], [1], [2]], device=system.device))
                ],
                properties=Labels(
                    "distance", torch.tensor([[0]], device=system.device)
                ),
            )
            system.add_neighbor_list(options, neighbors)

    return systems
