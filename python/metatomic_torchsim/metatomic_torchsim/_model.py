"""TorchSim wrapper for metatomic atomistic models.

Adapts metatomic models to the TorchSim ModelInterface protocol, allowing them to
be used within the torch-sim simulation framework for MD and other simulations.

Supports batched computations for multiple systems simultaneously, computing
energies, forces, and stresses via autograd.  Also supports output variants,
non-conservative forces/stress, energy uncertainty warnings, and additional
model outputs.
"""

import logging
import os
import pathlib
import warnings
from typing import Dict, List, Optional, Union

import torch
from metatensor.torch import TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
    pick_device,
    pick_output,
)

from ._neighbors import _compute_requested_neighbors


try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
except ImportError as e:
    raise ImportError(
        "the torch_sim package is required for metatomic-torchsim: "
        "pip install torch-sim-atomistic"
    ) from e


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
        variants: Optional[Dict[str, Optional[str]]] = None,
        non_conservative: bool = False,
        uncertainty_threshold: Optional[float] = 0.1,
        additional_outputs: Optional[Dict[str, ModelOutput]] = None,
    ) -> None:
        """
        :param model: Model to use.  Accepts a file path to a ``.pt`` saved
            model, a Python :py:class:`AtomisticModel` instance, or a
            TorchScript :py:class:`torch.jit.RecursiveScriptModule`.
        :param extensions_directory: Directory containing compiled TorchScript
            extensions required by the model, if any.
        :param device: Torch device for evaluation.  When ``None``, the best
            device is selected from the model's ``supported_devices``.
        :param check_consistency: Run consistency checks during model evaluation.
            Useful for debugging but hurts performance.
        :param compute_forces: Compute atomic forces via autograd.
        :param compute_stress: Compute stress tensors via the strain trick.
        :param variants: Dictionary mapping output names to a variant that should
            be used.  Setting ``{"energy": "pbe"}`` selects the ``"energy/pbe"``
            output.  The energy variant propagates to uncertainty and
            non-conservative outputs unless overridden (e.g.
            ``{"energy": "pbe", "energy_uncertainty": "r2scan"}`` would select
            ``energy/pbe`` and ``energy_uncertainty/r2scan``).
        :param non_conservative: If ``True``, the model will be asked to compute
            non-conservative forces and stresses.  This can afford a speed-up,
            potentially at the expense of physical correctness (especially in
            molecular dynamics simulations).
        :param uncertainty_threshold: Threshold for per-atom energy uncertainty
            in eV.  When the model supports ``energy_uncertainty`` with
            ``per_atom=True``, atoms exceeding this threshold trigger a warning.
            Set to ``None`` to disable.
        :param additional_outputs: Dictionary of extra :py:class:`ModelOutput`
            to request from the model.  Results are stored in
            :py:attr:`additional_outputs` after each forward call.
        """
        super().__init__()

        self._check_consistency = check_consistency

        # Load the model, following the same patterns as ase_calculator.py
        if isinstance(model, (str, bytes, pathlib.PurePath)):
            model_path = str(model)
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

        # Resolve output keys based on requested variants
        variants = variants or {}
        default_variant = variants.get("energy")

        resolved_variants = {
            key: variants.get(key, default_variant)
            for key in [
                "energy",
                "energy_uncertainty",
                "non_conservative_forces",
                "non_conservative_stress",
            ]
        }

        outputs = capabilities.outputs

        has_energy = any(
            "energy" == key or key.startswith("energy/") for key in outputs.keys()
        )
        if not has_energy:
            raise ValueError(
                "model does not have an 'energy' output. "
                "Only models with energy outputs can be used with TorchSim."
            )

        self._energy_key = pick_output("energy", outputs, resolved_variants["energy"])

        # Uncertainty
        has_energy_uq = any("energy_uncertainty" in key for key in outputs.keys())
        if has_energy_uq and uncertainty_threshold is not None:
            self._energy_uq_key = pick_output(
                "energy_uncertainty",
                outputs,
                resolved_variants["energy_uncertainty"],
            )
        else:
            self._energy_uq_key = None

        # Non-conservative outputs
        self._non_conservative = non_conservative
        if non_conservative:
            if (
                "non_conservative_stress" in variants
                and "non_conservative_forces" in variants
                and (
                    (variants["non_conservative_stress"] is None)
                    != (variants["non_conservative_forces"] is None)
                )
            ):
                raise ValueError(
                    "if both 'non_conservative_stress' and "
                    "'non_conservative_forces' are present in `variants`, they "
                    "must either be both `None` or both not `None`."
                )

            self._nc_forces_key = pick_output(
                "non_conservative_forces",
                outputs,
                resolved_variants["non_conservative_forces"],
            )
            self._nc_stress_key = pick_output(
                "non_conservative_stress",
                outputs,
                resolved_variants["non_conservative_stress"],
            )
        else:
            self._nc_forces_key = None
            self._nc_stress_key = None

        # Additional outputs
        if additional_outputs is None:
            self._additional_output_requests: Dict[str, ModelOutput] = {}
        else:
            assert isinstance(additional_outputs, dict)
            for name, output in additional_outputs.items():
                assert isinstance(name, str)
                assert isinstance(output, torch.ScriptObject), (
                    "outputs must be ModelOutput instances"
                )
            self._additional_output_requests = additional_outputs

        self._model = model.to(device=self._device)
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._uncertainty_threshold = uncertainty_threshold

        self._calculate_uncertainty = (
            self._energy_uq_key in outputs
            and outputs[self._energy_uq_key].per_atom
            and uncertainty_threshold is not None
        )

        if self._calculate_uncertainty:
            if uncertainty_threshold <= 0.0:
                raise ValueError(
                    f"`uncertainty_threshold` is {uncertainty_threshold} but must "
                    "be positive"
                )

        self._requested_neighbor_lists = self._model.requested_neighbor_lists()
        self._requested_inputs = self._model.requested_inputs()
        if len(self._requested_inputs) != 0:
            raise ValueError(
                "this model requests extra inputs "
                f"({', '.join(self._requested_inputs.keys())}), which are not "
                "implemented in metatomic-torchsim. Please open an issue if "
                "you need them!"
            )

        # Precompute the outputs dict (immutable after __init__)
        run_outputs: Dict[str, ModelOutput] = {
            self._energy_key: ModelOutput(quantity="energy", unit="eV", per_atom=False),
        }
        if self._calculate_uncertainty:
            run_outputs[self._energy_uq_key] = ModelOutput(
                quantity="energy", unit="eV", per_atom=True
            )
        if self._non_conservative:
            if self._compute_forces:
                run_outputs[self._nc_forces_key] = ModelOutput(
                    quantity="force", unit="eV/Angstrom", per_atom=True
                )
            if self._compute_stress:
                run_outputs[self._nc_stress_key] = ModelOutput(
                    quantity="pressure", unit="eV/Angstrom^3", per_atom=False
                )
        run_outputs.update(self._additional_output_requests)

        self._evaluation_options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs=run_outputs,
        )

        self.additional_outputs: Dict[str, TensorMap] = {}
        """
        Additional outputs computed by :py:meth:`forward` are stored here.
        Keys match the ``additional_outputs`` parameter to the constructor;
        values are raw :py:class:`metatensor.torch.TensorMap` from the model.
        """

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

        # Determine whether autograd is needed
        do_autograd_forces = self._compute_forces and not self._non_conservative
        do_autograd_stress = self._compute_stress and not self._non_conservative

        # Build per-system System objects.  Metatomic expects a list of System
        # rather than a single batched graph.
        systems: List[System] = []
        strains: List[torch.Tensor] = []
        n_systems = len(cell)

        pbc = state.pbc
        if isinstance(pbc, bool):
            pbc = torch.tensor([pbc, pbc, pbc])
        elif not isinstance(pbc, torch.Tensor):
            pbc = torch.tensor(pbc)

        for sys_idx in range(n_systems):
            mask = state.system_idx == sys_idx
            sys_positions = positions[mask]
            sys_cell = cell[sys_idx]
            sys_types = atomic_nums[mask]

            if do_autograd_forces:
                sys_positions = sys_positions.detach().requires_grad_(True)

            if do_autograd_stress:
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
                    pbc=pbc,
                )
            )

        # Compute neighbor lists
        systems = _compute_requested_neighbors(
            systems=systems,
            requested_options=self._requested_neighbor_lists,
            check_consistency=self._check_consistency,
        )

        # Run the model (evaluation options precomputed in __init__)
        model_outputs = self._model(
            systems=systems,
            options=self._evaluation_options,
            check_consistency=self._check_consistency,
        )

        energy_values = model_outputs[self._energy_key].block().values

        results: Dict[str, torch.Tensor] = {}
        results["energy"] = energy_values.detach().squeeze(-1)

        # Uncertainty warning
        if self._calculate_uncertainty:
            uncertainty = model_outputs[self._energy_uq_key].block().values
            n_total_atoms = positions.shape[0]
            if uncertainty.shape != (n_total_atoms, 1):
                raise ValueError(
                    f"expected uncertainty shape ({n_total_atoms}, 1), "
                    f"got {uncertainty.shape}"
                )
            threshold = self._uncertainty_threshold
            if torch.any(uncertainty > threshold):
                exceeded = torch.where(uncertainty.squeeze(-1) > threshold)[0]
                atom_list = exceeded.tolist()
                if len(atom_list) > 20:
                    atom_list = atom_list[:20]
                    suffix = f" (and {len(exceeded) - 20} more)"
                else:
                    suffix = ""
                warnings.warn(
                    "Some of the atomic energy uncertainties are larger than the "
                    f"threshold of {threshold} eV. The prediction is above the "
                    f"threshold for atoms {atom_list}{suffix}.",
                    stacklevel=2,
                )

        # Forces and stresses
        if self._non_conservative:
            if self._compute_forces:
                nc_forces = model_outputs[self._nc_forces_key].block().values.detach()
                nc_forces = nc_forces.reshape(-1, 3)
                # Remove spurious net force per system
                for sys_idx in range(n_systems):
                    mask = state.system_idx == sys_idx
                    sys_forces = nc_forces[mask]
                    nc_forces[mask] = sys_forces - sys_forces.mean(dim=0, keepdim=True)
                results["forces"] = nc_forces

            if self._compute_stress:
                nc_stress = model_outputs[self._nc_stress_key].block().values.detach()
                nc_stress = nc_stress.reshape(n_systems, 3, 3)
                results["stress"] = nc_stress

        elif do_autograd_forces or do_autograd_stress:
            grad_inputs: List[torch.Tensor] = []
            if do_autograd_forces:
                for system in systems:
                    grad_inputs.append(system.positions)
            if do_autograd_stress:
                grad_inputs.extend(strains)

            grads = torch.autograd.grad(
                outputs=energy_values,
                inputs=grad_inputs,
                grad_outputs=torch.ones_like(energy_values),
            )

            if do_autograd_forces and do_autograd_stress:
                n_sys = len(systems)
                force_grads = grads[:n_sys]
                stress_grads = grads[n_sys:]
            elif do_autograd_forces:
                force_grads = grads
                stress_grads = ()
            else:
                force_grads = ()
                stress_grads = grads

            if do_autograd_forces:
                results["forces"] = torch.cat([-g for g in force_grads])

            if do_autograd_stress:
                results["stress"] = torch.stack(
                    [
                        g / torch.abs(torch.det(system.cell.detach()))
                        for g, system in zip(stress_grads, systems, strict=True)
                    ]
                )

        # Store additional outputs
        self.additional_outputs = {}
        for name in self._additional_output_requests:
            self.additional_outputs[name] = model_outputs[name]

        return results
