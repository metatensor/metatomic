"""TorchSim wrapper for metatomic atomistic models.

Adapts metatomic models to the TorchSim ModelInterface protocol, allowing them to
be used within the torch-sim simulation framework for MD and other simulations.

Supports batched computations for multiple systems simultaneously, computing
energies, forces, and stresses via autograd.
"""

import logging
import os
import pathlib
from typing import Dict, List, Optional, Union

import torch
import vesin.metatomic
from metatensor.torch import Labels, TensorBlock

from metatomic.torch import (
    AtomisticModel,
    ModelEvaluationOptions,
    ModelOutput,
    NeighborListOptions,
    System,
    load_atomistic_model,
    pick_device,
)


try:
    from nvalchemiops.neighborlist import neighbor_list as nvalchemi_neighbor_list

    HAS_NVALCHEMIOPS = True
except ImportError:
    HAS_NVALCHEMIOPS = False


try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
except ImportError as exc:
    raise ImportError(
        "torch-sim is required for metatomic-torchsim: pip install torch-sim-atomistic"
    ) from exc


FilePath = Union[str, bytes, pathlib.PurePath]

LOGGER = logging.getLogger(__name__)

STR_TO_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class MetatomicModel(ModelInterface):
    """TorchSim wrapper for metatomic atomistic models.

    Wraps a metatomic model to compute energies, forces, and stresses within the
    TorchSim framework.  Handles the translation between TorchSim's batched
    ``SimState`` and metatomic's list-of-``System`` convention, and uses autograd
    for force/stress derivatives.

    Neighbor lists are computed with vesin, or with nvalchemiops on CUDA when
    available and the model requests full neighbor lists.
    """

    def __init__(
        self,
        model: Union[FilePath, AtomisticModel, "torch.jit.RecursiveScriptModule"],
        *,
        extensions_directory: Optional[FilePath] = None,
        device: Optional[Union[torch.device, str]] = None,
        check_consistency: bool = False,
        compute_forces: bool = True,
        compute_stress: bool = True,
    ) -> None:
        """Initialize the metatomic model wrapper.

        :param model: Model to use.  Accepts a file path to a ``.pt`` saved
            model, a ``.ckpt`` metatrain checkpoint (requires ``metatrain``), the
            string ``"pet-mad"`` (shortcut for the PET-MAD model, requires
            ``metatrain``), a Python :py:class:`AtomisticModel` instance, or a
            TorchScript :py:class:`torch.jit.RecursiveScriptModule`.
        :param extensions_directory: Directory containing compiled TorchScript
            extensions required by the model, if any.
        :param device: Torch device for evaluation.  When ``None``, the best
            device is selected from the model's ``supported_devices``.
        :param check_consistency: Run consistency checks during model evaluation.
            Useful for debugging but hurts performance.
        :param compute_forces: Compute atomic forces via autograd.
        :param compute_stress: Compute stress tensors via the strain trick.
        """
        super().__init__()

        self._check_consistency = check_consistency

        # Load the model, following the same patterns as ase_calculator.py
        if isinstance(model, str) and model == "pet-mad":
            model = self._load_metatrain_model(
                "https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/"
                "models/pet-mad-v1.1.0.ckpt"
            )
        elif isinstance(model, (str, bytes, pathlib.PurePath)):
            model_path = str(model)
            if model_path.endswith(".ckpt"):
                model = self._load_metatrain_model(model_path)
            else:
                if not os.path.exists(model_path):
                    raise ValueError(f"given model path '{model_path}' does not exist")
                model = load_atomistic_model(
                    model_path, extensions_directory=extensions_directory
                )
        elif isinstance(model, torch.jit.RecursiveScriptModule):
            if model.original_name != "AtomisticModel":
                raise TypeError(
                    "torch model must be 'AtomisticModel', "
                    f"got '{model.original_name}' instead"
                )
        elif isinstance(model, AtomisticModel):
            pass
        else:
            raise TypeError(f"unknown type for model: {type(model)}")

        capabilities = model.capabilities()

        # Resolve device
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            self._device = device
        else:
            self._device = torch.device(
                pick_device(capabilities.supported_devices, None)
            )

        # Resolve dtype from model capabilities
        if capabilities.dtype in STR_TO_DTYPE:
            self._dtype = STR_TO_DTYPE[capabilities.dtype]
        else:
            raise ValueError(
                f"unexpected dtype in model capabilities: {capabilities.dtype}"
            )

        if "energy" not in capabilities.outputs:
            raise ValueError(
                "model does not have an 'energy' output. "
                "Only models with energy outputs can be used with TorchSim."
            )

        self._model = model.to(device=self._device)
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"
        self._requested_neighbor_lists = self._model.requested_neighbor_lists()

        self._evaluation_options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs={
                "energy": ModelOutput(quantity="energy", unit="eV", per_atom=False)
            },
        )

    @staticmethod
    def _load_metatrain_model(path: str) -> AtomisticModel:
        """Load a metatrain checkpoint and export it as an AtomisticModel."""
        try:
            from metatrain.utils.io import load_model
        except ImportError as exc:
            raise ImportError(
                "metatrain is required to load .ckpt files or use the 'pet-mad' "
                "shortcut: pip install metatrain"
            ) from exc

        return load_model(path).export()

    def forward(self, state: "ts.SimState") -> Dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the given simulation state.

        :param state: TorchSim simulation state

        :returns: Dictionary with ``"energy"`` (shape ``[n_systems]``),
            ``"forces"`` (shape ``[n_atoms, 3]``, if ``compute_forces``), and
            ``"stress"`` (shape ``[n_systems, 3, 3]``, if ``compute_stress``).
        """
        positions = state.positions
        cell = state.row_vector_cell
        atomic_nums = state.atomic_numbers

        if positions.dtype != self._dtype:
            raise TypeError(
                f"positions dtype {positions.dtype} does not match "
                f"model dtype {self._dtype}"
            )

        # Build per-system System objects.  Metatomic expects a list of System
        # rather than a single batched graph.
        systems: List[System] = []
        strains: List[torch.Tensor] = []
        n_systems = len(cell)

        for sys_idx in range(n_systems):
            mask = state.system_idx == sys_idx
            sys_positions = positions[mask]
            sys_cell = cell[sys_idx]
            sys_types = atomic_nums[mask]

            if self._compute_forces:
                sys_positions = sys_positions.detach().requires_grad_(True)

            if self._compute_stress:
                strain = torch.eye(
                    3,
                    device=self._device,
                    dtype=self._dtype,
                    requires_grad=True,
                )
                sys_positions = sys_positions @ strain
                sys_cell = sys_cell @ strain
                strains.append(strain)

            systems.append(
                System(
                    positions=sys_positions,
                    types=sys_types,
                    cell=sys_cell,
                    pbc=state.pbc,
                )
            )

        # Compute neighbor lists
        systems = _compute_requested_neighbors(
            systems=systems,
            requested_options=self._requested_neighbor_lists,
            check_consistency=self._check_consistency,
        )

        # Run the model
        model_outputs = self._model(
            systems=systems,
            options=self._evaluation_options,
            check_consistency=self._check_consistency,
        )

        energy_values = model_outputs["energy"].block().values

        results: Dict[str, torch.Tensor] = {}
        results["energy"] = energy_values.detach().squeeze(-1)

        # Compute forces and/or stresses via autograd
        if self._compute_forces or self._compute_stress:
            grad_inputs: List[torch.Tensor] = []
            if self._compute_forces:
                for system in systems:
                    grad_inputs.append(system.positions)
            if self._compute_stress:
                grad_inputs.extend(strains)

            grads = torch.autograd.grad(
                outputs=energy_values,
                inputs=grad_inputs,
                grad_outputs=torch.ones_like(energy_values),
            )

            if self._compute_forces and self._compute_stress:
                n_sys = len(systems)
                force_grads = grads[:n_sys]
                stress_grads = grads[n_sys:]
            elif self._compute_forces:
                force_grads = grads
                stress_grads = ()
            else:
                force_grads = ()
                stress_grads = grads

            if self._compute_forces:
                results["forces"] = torch.cat([-g for g in force_grads])

            if self._compute_stress:
                results["stress"] = torch.stack(
                    [
                        g / torch.abs(torch.det(system.cell.detach()))
                        for g, system in zip(stress_grads, systems, strict=False)
                    ]
                )

        return results


# -- Neighbor list helpers (shared with ase_calculator.py patterns) ----------


def _compute_requested_neighbors(
    systems: List[System],
    requested_options: List[NeighborListOptions],
    check_consistency: bool = False,
) -> List[System]:
    """Compute all neighbor lists requested by the model and store them in the systems.

    Uses nvalchemiops for full neighbor lists on CUDA when available, vesin otherwise.
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
    check_consistency: bool = False,
) -> List[System]:
    """Compute neighbor lists using vesin."""
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


def _compute_requested_neighbors_nvalchemi(
    systems: List[System],
    requested_options: List[NeighborListOptions],
) -> List[System]:
    """Compute full neighbor lists on CUDA using nvalchemiops."""
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
                    Labels(
                        "xyz",
                        torch.tensor([[0], [1], [2]], device=system.device),
                    )
                ],
                properties=Labels(
                    "distance",
                    torch.tensor([[0]], device=system.device),
                ),
            )
            system.add_neighbor_list(options, neighbors)

    return systems
