import warnings
from typing import List

import torch
import vesin.metatomic
from metatensor.torch import Labels, TensorBlock

from metatomic.torch import NeighborListOptions, System


try:
    import nvalchemiops

    NVALCHEMI_MIN = "0.3.0"
    NVALCHEMI_MAX = "0.4.0"

    if (
        nvalchemiops.__version__ < NVALCHEMI_MIN
        or nvalchemiops.__version__ >= NVALCHEMI_MAX
    ):
        warnings.warn(
            f"found nvalchemi-toolkit-ops version {nvalchemiops.__version__}, this "
            f"code was only tested with version >={NVALCHEMI_MIN},<={NVALCHEMI_MAX}. "
            "If you encounter errors, please update to this version range.",
            stacklevel=1,
        )
    from nvalchemiops.torch.neighbors import neighbor_list as nvalchemi_neighbor_list

    HAS_NVALCHEMIOPS = True
except ImportError:
    HAS_NVALCHEMIOPS = False


class AllNeighborsCalculator:
    def __init__(
        self,
        requested_options: List[NeighborListOptions],
        check_consistency=False,
    ):
        self.check_consistency = check_consistency
        self._full_nl_options = [
            options for options in requested_options if options.full_list
        ]
        self._full_vesin_calculators = [
            vesin.metatomic.NeighborList(
                options=options,
                length_unit="angstrom",
                check_consistency=check_consistency,
            )
            for options in requested_options
            if options.full_list
        ]
        self._half_vesin_calculators = [
            vesin.metatomic.NeighborList(
                options=options,
                length_unit="angstrom",
                check_consistency=check_consistency,
            )
            for options in requested_options
            if not options.full_list
        ]

    def compute(self, systems: List[System]) -> List[System]:
        assert isinstance(systems, list)
        assert isinstance(systems[0], torch.ScriptObject)

        can_use_nvalchemi = HAS_NVALCHEMIOPS and all(
            system.device.type == "cuda" for system in systems
        )

        if can_use_nvalchemi:
            # Do the full neighbor lists with nvalchemi
            systems = _compute_requested_neighbors_nvalchemi(
                systems=systems,
                requested_options=self._full_nl_options,
            )
        else:
            systems = _compute_requested_neighbors_vesin(
                systems=systems,
                calculators=self._full_vesin_calculators,
            )

        # always compute the half neighbor lists with vesin
        systems = _compute_requested_neighbors_vesin(
            systems=systems,
            calculators=self._half_vesin_calculators,
        )

        return systems


def _compute_requested_neighbors_vesin(
    systems: List[System],
    calculators: List[vesin.metatomic.NeighborList],
) -> List[System]:
    system_devices = []
    moved_systems = []
    for system in systems:
        system_devices.append(system.device)
        if system.device.type not in ["cpu", "cuda"]:
            moved_systems.append(system.to(device="cpu"))
        else:
            moved_systems.append(system)

    for calculator in calculators:
        calculator.add_neighbor_list(
            systems=moved_systems,
            # if we have more than one system, we can no keep the data as a reference
            # to memory allocated in the calculator and we need to make a copy
            copy=len(systems) > 1,
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
                max_neighbors=16384,
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
                    assume_unique=True,
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
