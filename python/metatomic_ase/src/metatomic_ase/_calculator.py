import logging
import os
import pathlib
import warnings
from typing import Dict, List, Optional, Union

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from torch.profiler import record_function

from metatomic.torch import (
    AtomisticModel,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    load_atomistic_model,
    pick_device,
    pick_output,
)

from ._neighbors import _compute_requested_neighbors


import ase  # isort: skip
import ase.calculators.calculator  # isort: skip
from ase.calculators.calculator import (  # isort: skip
    InputError,
    PropertyNotImplementedError,
    all_properties as ALL_ASE_PROPERTIES,
)


FilePath = Union[str, bytes, pathlib.PurePath]

LOGGER = logging.getLogger(__name__)


STR_TO_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _get_charges(atoms: ase.Atoms) -> np.ndarray:
    try:
        return atoms.get_charges()
    except Exception:
        return atoms.get_initial_charges()


ARRAY_QUANTITIES = {
    "momenta": {
        "quantity": "momentum",
        "getter": ase.Atoms.get_momenta,
        "unit": "(eV*u)^(1/2)",
    },
    "masses": {
        "quantity": "mass",
        "getter": ase.Atoms.get_masses,
        "unit": "u",
    },
    "velocities": {
        "quantity": "velocity",
        "getter": ase.Atoms.get_velocities,
        "unit": "(eV/u)^(1/2)",
    },
    "charges": {
        "quantity": "charge",
        "getter": _get_charges,
        "unit": "e",
    },
    "ase::initial_magmoms": {
        "quantity": "magnetic_moment",
        "getter": ase.Atoms.get_initial_magnetic_moments,
        "unit": "",
    },
    "ase::magnetic_moment": {
        "quantity": "magnetic_moment",
        "getter": ase.Atoms.get_magnetic_moment,
        "unit": "",
    },
    "ase::magnetic_moments": {
        "quantity": "magnetic_moment",
        "getter": ase.Atoms.get_magnetic_moments,
        "unit": "",
    },
    "ase::initial_charges": {
        "quantity": "charge",
        "getter": ase.Atoms.get_initial_charges,
        "unit": "e",
    },
    "ase::dipole_moment": {
        "quantity": "dipole_moment",
        "getter": ase.Atoms.get_dipole_moment,
        "unit": "",
    },
}


class MetatomicCalculator(ase.calculators.calculator.Calculator):
    """
    The :py:class:`MetatomicCalculator` class implements ASE's
    :py:class:`ase.calculators.calculator.Calculator` API using metatomic models to
    compute energy, forces and any other supported property.

    This class can be initialized with any :py:class:`metatomic.torch.AtomisticModel`,
    and used to run simulations using ASE's MD facilities.

    Neighbor lists are computed using the fast `vesin
    <https://luthaf.fr/vesin/latest/index.html>`_ neighbor list library, either on CPU
    or GPU depending on the device of the model. If `nvalchemiops
    <https://github.com/NVIDIA/nvalchemi-toolkit-ops>`_ is installed, full neighbor
    lists on GPU will be computed with it instead.
    """

    def __init__(
        self,
        model: Union[FilePath, AtomisticModel],
        *,
        additional_outputs: Optional[Dict[str, ModelOutput]] = None,
        extensions_directory=None,
        check_consistency=False,
        device=None,
        variants: Optional[Dict[str, Optional[str]]] = None,
        non_conservative=False,
        do_gradients_with_energy=True,
        uncertainty_threshold=0.1,
    ):
        """
        :param model: model to use for the calculation. This can be a file path, a
            Python instance of :py:class:`metatomic.torch.AtomisticModel`, or the output
            of :py:func:`torch.jit.script` on
            :py:class:`metatomic.torch.AtomisticModel`.
        :param additional_outputs: Dictionary of additional outputs to be computed by
            the model. These outputs will always be computed whenever the
            :py:meth:`calculate` function is called (e.g. by
            :py:meth:`ase.Atoms.get_potential_energy`,
            :py:meth:`ase.optimize.optimize.Dynamics.run`, *etc.*) and stored in the
            :py:attr:`additional_outputs` attribute. If you want more control over when
            and how to compute specific outputs, you should use :py:meth:`run_model`
            instead.
        :param extensions_directory: if the model uses extensions, we will try to load
            them from this directory
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        :param device: torch device to use for the calculation. If ``None``, we will try
            the options in the model's ``supported_device`` in order.
        :param variants: dictionary mapping output names to a variant that should be
            used for the calculations (e.g. ``{"energy": "PBE"}``). If ``"energy"`` is
            set to a variant also the uncertainty and non conservative outputs will be
            taken from this variant. This behaviour can be overriden by setting the
            corresponding keys explicitly to ``None`` or to another value (e.g.
            ``{"energy_uncertainty": "r2scan"}``).
        :param non_conservative: if ``True``, the model will be asked to compute
            non-conservative forces and stresses. This can afford a speed-up,
            potentially at the expense of physical correctness (especially in molecular
            dynamics simulations).
        :param do_gradients_with_energy: if ``True``, this calculator will always
            compute the energy gradients (forces and stress) when the energy is
            requested (e.g. through ``atoms.get_potential_energy()``). Because the
            results of a calculation are cached by ASE, this means future calls to
            ``atom.get_forces()`` will return immediately, without needing to execute
            the model again. If you are mainly interested in the energy, you can set
            this to ``False`` and enjoy a faster model. Forces will still be calculated
            if requested with ``atoms.get_forces()``.
        :param uncertainty_threshold: threshold for the atomic energy uncertainty in eV.
            This will only be used if the model supports atomic uncertainty estimation
            (https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c00704). Set this to
            ``None`` to disable uncertainty quantification even if the model supports
            it.
        """
        super().__init__()

        self.parameters = {
            "extensions_directory": extensions_directory,
            "check_consistency": bool(check_consistency),
            "variants": variants,
            "non_conservative": bool(non_conservative),
            "do_gradients_with_energy": bool(do_gradients_with_energy),
            "additional_outputs": additional_outputs,
            "uncertainty_threshold": uncertainty_threshold,
        }

        # Load the model
        if isinstance(model, (str, bytes, pathlib.PurePath)):
            if not os.path.exists(model):
                raise InputError(f"given model path '{model}' does not exist")

            # only store the model in self.parameters if is it the path to a file
            self.parameters["model"] = str(model)

            model = load_atomistic_model(
                model, extensions_directory=extensions_directory
            )

        elif isinstance(model, torch.jit.RecursiveScriptModule):
            if model.original_name != "AtomisticModel":
                raise InputError(
                    "torch model must be 'AtomisticModel', "
                    f"got '{model.original_name}' instead"
                )
        elif isinstance(model, AtomisticModel):
            # nothing to do
            pass
        else:
            raise TypeError(f"unknown type for model: {type(model)}")

        self.parameters["device"] = str(device) if device is not None else None
        # get the best device according what the model supports and what's available on
        # the current machine
        capabilities = model.capabilities()
        self._device = torch.device(
            pick_device(capabilities.supported_devices, self.parameters["device"])
        )

        if capabilities.dtype in STR_TO_DTYPE:
            self._dtype = STR_TO_DTYPE[capabilities.dtype]
        else:
            raise ValueError(
                f"found unexpected dtype in model capabilities: {capabilities.dtype}"
            )

        # resolve the output keys to use based on the requested variants
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

        # Check if the model has an energy output
        has_energy = any(
            "energy" == key or key.startswith("energy/") for key in outputs.keys()
        )
        if has_energy:
            self._energy_key = pick_output(
                "energy", outputs, resolved_variants["energy"]
            )
        else:
            self._energy_key = None

        has_energy_uq = any("energy_uncertainty" in key for key in outputs.keys())
        if has_energy_uq and uncertainty_threshold is not None:
            self._energy_uq_key = pick_output(
                "energy_uncertainty", outputs, resolved_variants["energy_uncertainty"]
            )
        else:
            self._energy_uq_key = "energy_uncertainty"

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
            self._nc_forces_key = "non_conservative_forces"
            self._nc_stress_key = "non_conservative_stress"

        if additional_outputs is None:
            self._additional_output_requests = {}
        else:
            assert isinstance(additional_outputs, dict)
            for name, output in additional_outputs.items():
                assert isinstance(name, str)
                assert isinstance(output, torch.ScriptObject)
                assert "explicit_gradients_setter" in output._method_names(), (
                    "outputs must be ModelOutput instances"
                )

            self._additional_output_requests = additional_outputs

        self._model = model.to(device=self._device)

        self._calculate_uncertainty = (
            self._energy_uq_key in self._model.capabilities().outputs
            # we require per-atom uncertainties to capture local effects
            and self._model.capabilities().outputs[self._energy_uq_key].per_atom
            and uncertainty_threshold is not None
        )

        if self._calculate_uncertainty:
            assert uncertainty_threshold is not None
            if uncertainty_threshold <= 0.0:
                raise ValueError(
                    f"`uncertainty_threshold` is {uncertainty_threshold} but must "
                    "be positive"
                )

        # We do our own check to verify if a property is implemented in `calculate()`,
        # so we pretend to be able to compute all properties ASE knows about.
        self.implemented_properties = ALL_ASE_PROPERTIES

        self.additional_outputs: Dict[str, TensorMap] = {}
        """
        Additional outputs computed by :py:meth:`calculate` are stored in this
        dictionary.

        The keys will match the keys of the ``additional_outputs`` parameters to the
        constructor; and the values will be the corresponding raw
        :py:class:`metatensor.torch.TensorMap` produced by the model.
        """

    def todict(self):
        if "model" not in self.parameters:
            raise RuntimeError(
                "can not save metatensor model in ASE `todict`, please initialize "
                "`MetatomicCalculator` with a path to a saved model file if you need "
                "to use `todict`"
            )

        return self.parameters

    @classmethod
    def fromdict(cls, data):
        return MetatomicCalculator(**data)

    def metadata(self) -> ModelMetadata:
        """Get the metadata of the underlying model"""
        return self._model.metadata()

    def run_model(
        self,
        atoms: Union[ase.Atoms, List[ase.Atoms]],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Run the model on the given ``atoms``, computing the requested ``outputs`` and
        only these.

        The output of the model is returned directly, and as such the blocks' ``values``
        will be :py:class:`torch.Tensor`.

        This is intended as an easy way to run metatensor models on
        :py:class:`ase.Atoms` when the model can compute outputs not supported by the
        standard ASE's calculator interface.

        All the parameters have the same meaning as the corresponding ones in
        :py:meth:`metatomic.torch.ModelInterface.forward`.

        :param atoms: :py:class:`ase.Atoms`, or list of :py:class:`ase.Atoms`, on which
            to run the model
        :param outputs: outputs of the model that should be predicted
        :param selected_atoms: subset of atoms on which to run the calculation
        """
        if isinstance(atoms, ase.Atoms):
            atoms_list = [atoms]
        else:
            atoms_list = atoms

        systems = []
        for atoms in atoms_list:
            types, positions, cell, pbc = _ase_to_torch_data(
                atoms=atoms, dtype=self._dtype, device=self._device
            )
            system = System(types, positions, cell, pbc)
            # Get the additional inputs requested by the model
            for name, option in self._model.requested_inputs().items():
                input_tensormap = _get_ase_input(
                    atoms, name, option, dtype=self._dtype, device=self._device
                )
                system.add_data(name, input_tensormap)
            systems.append(system)

        # Compute the neighbors lists requested by the model
        input_systems = _compute_requested_neighbors(
            systems=systems,
            requested_options=self._model.requested_neighbor_lists(),
            check_consistency=self.parameters["check_consistency"],
        )

        available_outputs = self._model.capabilities().outputs
        for key in outputs:
            if key not in available_outputs:
                raise ValueError(f"this model does not support '{key}' output")

        options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs=outputs,
            selected_atoms=selected_atoms,
        )
        return self._model(
            systems=input_systems,
            options=options,
            check_consistency=self.parameters["check_consistency"],
        )

    def calculate(
        self,
        atoms: ase.Atoms,
        properties: List[str],
        system_changes: List[str],
    ) -> None:
        """
        Compute some ``properties`` with this calculator, and return them in the format
        expected by ASE.

        This is not intended to be called directly by users, but to be an implementation
        detail of ``atoms.get_energy()`` and related functions. See
        :py:meth:`ase.calculators.calculator.Calculator.calculate` for more information.
        """
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        # In the next few lines, we decide which properties to calculate among energy,
        # forces and stress. In addition to the requested properties, we calculate the
        # energy if any of the three is requested, as it is an intermediate step in the
        # calculation of the other two. We also calculate the forces if the stress is
        # requested, and vice-versa. The overhead for the latter operation is also
        # small, assuming that the majority of the model computes forces and stresses
        # by backward propagation as opposed to forward-mode differentiation.
        calculate_energy = (
            "energy" in properties
            or "energies" in properties
            or "forces" in properties
            or "stress" in properties
        )

        # Check if energy-related properties are requested but the model doesn't
        # support energy
        if calculate_energy and self._energy_key is None:
            raise PropertyNotImplementedError(
                "This calculator does not support energy-related properties "
                "(energy, energies, forces, stress) because the underlying model "
                "does not have an energy output"
            )
        calculate_energies = "energies" in properties
        calculate_forces = "forces" in properties or "stress" in properties
        calculate_stress = "stress" in properties
        if calculate_stress and not atoms.pbc.all():
            warnings.warn(
                "stress requested but likely to be wrong, since the system is not "
                "periodic in all directions",
                stacklevel=2,
            )
        if "forces" in properties and atoms.pbc.all():
            # we have PBCs, and, since the user/integrator requested forces, we will run
            # backward anyway, so let's do the stress as well for free (this saves
            # another forward-backward call later if the stress is requested)
            calculate_stress = True
        if "stresses" in properties:
            raise NotImplementedError("'stresses' are not implemented yet")

        if self.parameters["do_gradients_with_energy"]:
            if calculate_energies or calculate_energy:
                calculate_forces = True
                calculate_stress = True

        with record_function("MetatomicCalculator::prepare_inputs"):
            outputs = self._ase_properties_to_metatensor_outputs(
                properties,
                calculate_forces=calculate_forces,
                calculate_stress=calculate_stress,
                calculate_stresses=False,
            )
            outputs.update(self._additional_output_requests)
            if calculate_energy and self._calculate_uncertainty:
                outputs[self._energy_uq_key] = ModelOutput(
                    quantity="energy",
                    unit="eV",
                    per_atom=True,
                    explicit_gradients=[],
                )

            capabilities = self._model.capabilities()
            for name in outputs.keys():
                if name not in capabilities.outputs:
                    raise ValueError(
                        f"you asked for the calculation of {name}, but this model "
                        "does not support it"
                    )

            types, positions, cell, pbc = _ase_to_torch_data(
                atoms=atoms, dtype=self._dtype, device=self._device
            )

            do_backward = False
            if calculate_forces and not self.parameters["non_conservative"]:
                do_backward = True
                positions.requires_grad_(True)

            if calculate_stress and not self.parameters["non_conservative"]:
                do_backward = True

                strain = torch.eye(
                    3, requires_grad=True, device=self._device, dtype=self._dtype
                )

                positions = positions @ strain
                positions.retain_grad()

                cell = cell @ strain

            run_options = ModelEvaluationOptions(
                length_unit="angstrom",
                outputs=outputs,
                selected_atoms=None,
            )

        with record_function("MetatomicCalculator::compute_neighbors"):
            # convert from ase.Atoms to metatomic.torch.System
            system = System(types, positions, cell, pbc)
            input_system = _compute_requested_neighbors(
                systems=[system],
                requested_options=self._model.requested_neighbor_lists(),
                check_consistency=self.parameters["check_consistency"],
            )[0]

        with record_function("MetatomicCalculator::get_model_inputs"):
            for name, option in self._model.requested_inputs().items():
                input_tensormap = _get_ase_input(
                    atoms, name, option, dtype=self._dtype, device=self._device
                )
                input_system.add_data(name, input_tensormap)

        # no `record_function` here, this will be handled by AtomisticModel
        outputs = self._model(
            [input_system],
            run_options,
            check_consistency=self.parameters["check_consistency"],
        )
        energy = outputs[self._energy_key]

        with record_function("MetatomicCalculator::sum_energies"):
            if run_options.outputs[self._energy_key].per_atom:
                assert len(energy) == 1
                assert energy.sample_names == ["system", "atom"]
                assert torch.all(energy.block().samples["system"] == 0)
                energies = energy
                assert energies.block().values.shape == (len(atoms), 1)

                energy = mts.sum_over_samples(energy, sample_names=["atom"])

            assert len(energy.block().gradients_list()) == 0
            assert energy.block().values.shape == (1, 1)

        with record_function("ASECalculator::uncertainty_warning"):
            if calculate_energy and self._calculate_uncertainty:
                uncertainty = outputs[self._energy_uq_key].block().values
                assert uncertainty.shape == (len(atoms), 1)
                uncertainty = uncertainty.detach().cpu().numpy()

                threshold = self.parameters["uncertainty_threshold"]
                if np.any(uncertainty > threshold):
                    warnings.warn(
                        "Some of the atomic energy uncertainties are larger than the "
                        f"threshold of {threshold} eV. The prediction is above the "
                        f"threshold for atoms {np.where(uncertainty > threshold)[0]}.",
                        stacklevel=2,
                    )

        if do_backward:
            if energy.block().values.grad_fn is None:
                # did the user actually request a gradient, or are we trying to
                # compute one just for efficiency?
                if "forces" in properties or "stress" in properties:
                    # the user asked for it, let it fail below
                    pass
                else:
                    # we added the calculation, let's remove it
                    do_backward = False
                    calculate_forces = False
                    calculate_stress = False

        with record_function("MetatomicCalculator::run_backward"):
            if do_backward:
                energy.block().values.backward()

        with record_function("MetatomicCalculator::convert_outputs"):
            self.results = {}

            if calculate_energies:
                energies_values = energies.block().values.detach().reshape(-1)
                atom_indexes = energies.block().samples.column("atom")
                result = torch.zeros_like(energies_values)
                result.index_add_(0, atom_indexes, energies_values)
                self.results["energies"] = result.cpu().double().numpy()

            if calculate_energy:
                energy_values = energy.block().values.detach()
                energy_values = energy_values.cpu().double()
                self.results["energy"] = energy_values.numpy()[0, 0]

            if calculate_forces:
                if self.parameters["non_conservative"]:
                    forces_values = outputs[self._nc_forces_key].block().values.detach()
                    # remove any spurious net force
                    forces_values = forces_values - forces_values.mean(
                        dim=0, keepdim=True
                    )
                else:
                    forces_values = -system.positions.grad
                forces_values = forces_values.reshape(-1, 3)
                forces_values = forces_values.cpu().double()
                self.results["forces"] = forces_values.numpy()

            if calculate_stress:
                if self.parameters["non_conservative"]:
                    stress_values = outputs[self._nc_stress_key].block().values.detach()
                else:
                    stress_values = strain.grad / atoms.cell.volume
                stress_values = stress_values.reshape(3, 3)
                stress_values = stress_values.cpu().double()
                self.results["stress"] = _full_3x3_to_voigt_6_stress(
                    stress_values.numpy()
                )

            self.additional_outputs = {}
            for name in self._additional_output_requests:
                self.additional_outputs[name] = outputs[name]

    def compute_energy(
        self,
        atoms: Union[ase.Atoms, List[ase.Atoms]],
        compute_forces_and_stresses: bool = False,
        *,
        per_atom: bool = False,
    ) -> Dict[str, Union[Union[float, np.ndarray], List[Union[float, np.ndarray]]]]:
        """
        Compute the energy of the given ``atoms``.

        Energies are computed in eV, forces in eV/Å, and stresses in 3x3 tensor format
        and in units of eV/Å^3.

        :param atoms: :py:class:`ase.Atoms`, or list of :py:class:`ase.Atoms`, on which
            to run the model
        :param compute_forces_and_stresses: if ``True``, the model will also compute
            forces and stresses. IMPORTANT: stresses will only be computed if all
            provided systems have periodic boundary conditions in all directions.
        :param per_atom: if ``True``, the per-atom energies will also be
            computed.
        :return: A dictionary with the computed properties. The dictionary will contain
            the ``energy`` as a float. If ``compute_forces_and_stresses`` is True,
            the ``forces`` and ``stress`` will also be included as numpy arrays.
            If ``per_atom`` is True, the ``energies`` key will also be present,
            containing the per-atom energies as a numpy array.
            In case of a list of :py:class:`ase.Atoms`, the dictionary values will
            instead be lists of the corresponding properties, in the same format.
        """
        if self._energy_key is None:
            raise ValueError(
                "This calculator does not support energy computation because "
                "the underlying model does not have an energy output"
            )

        if isinstance(atoms, ase.Atoms):
            atoms_list = [atoms]
            was_single = True
        else:
            atoms_list = atoms
            was_single = False

        properties = ["energy"]
        energy_per_atom = False
        if per_atom:
            energy_per_atom = True
            properties.append("energies")

        outputs = self._ase_properties_to_metatensor_outputs(
            properties=properties,
            calculate_forces=compute_forces_and_stresses,
            calculate_stress=compute_forces_and_stresses,
            calculate_stresses=False,
        )

        systems = []
        if compute_forces_and_stresses:
            strains = []
        for atoms in atoms_list:
            types, positions, cell, pbc = _ase_to_torch_data(
                atoms=atoms, dtype=self._dtype, device=self._device
            )
            if compute_forces_and_stresses and not self.parameters["non_conservative"]:
                positions.requires_grad_(True)
                strain = torch.eye(
                    3, requires_grad=True, device=self._device, dtype=self._dtype
                )
                positions = positions @ strain
                positions.retain_grad()
                cell = cell @ strain
                strains.append(strain)
            system = System(types, positions, cell, pbc)
            systems.append(system)

        # Compute the neighbors lists requested by the model
        input_systems = _compute_requested_neighbors(
            systems=systems,
            requested_options=self._model.requested_neighbor_lists(),
            check_consistency=self.parameters["check_consistency"],
        )

        predictions = self._model(
            systems=input_systems,
            options=ModelEvaluationOptions(length_unit="angstrom", outputs=outputs),
            check_consistency=self.parameters["check_consistency"],
        )
        energies = predictions[self._energy_key]

        if energy_per_atom:
            # Get per-atom energies
            sorted_block = mts.sort_block(energies.block(), axes="samples")
            energies_values = sorted_block.values.detach().reshape(-1)

            split_sizes = [len(system) for system in systems]
            atom_indices = sorted_block.samples.column("atom")
            energies_values = torch.split(energies_values, split_sizes, dim=0)
            split_atom_indices = torch.split(atom_indices, split_sizes, dim=0)
            split_energies = []
            for atom_indices, values in zip(
                split_atom_indices, energies_values, strict=True
            ):
                split_energy = torch.zeros(
                    len(atom_indices), dtype=values.dtype, device=values.device
                )
                split_energy.index_add_(0, atom_indices, values)
                split_energies.append(split_energy)

            total_energy = (
                mts.sum_over_samples(energies, ["atom"])
                .block()
                .values.detach()
                .cpu()
                .double()
                .numpy()
                .flatten()
                .tolist()
            )
            results_as_numpy_arrays = {
                "energy": total_energy,
                "energies": [e.cpu().double().numpy() for e in split_energies],
            }
        else:
            results_as_numpy_arrays = {
                "energy": energies.block()
                .values.squeeze(-1)
                .detach()
                .cpu()
                .double()
                .numpy(),
            }

        if compute_forces_and_stresses:
            if self.parameters["non_conservative"]:
                results_as_numpy_arrays["forces"] = (
                    predictions[self._nc_forces_key]
                    .block()
                    .values.squeeze(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                # all the forces are concatenated in a single array, so we need to
                # split them into the original systems
                split_sizes = [len(system) for system in systems]
                split_indices = np.cumsum(split_sizes[:-1])
                results_as_numpy_arrays["forces"] = np.split(
                    results_as_numpy_arrays["forces"], split_indices, axis=0
                )

                # remove net forces
                results_as_numpy_arrays["forces"] = [
                    f - f.mean(axis=0, keepdims=True)
                    for f in results_as_numpy_arrays["forces"]
                ]

                if all(atoms.pbc.all() for atoms in atoms_list):
                    results_as_numpy_arrays["stress"] = [
                        s
                        for s in predictions[self._nc_stress_key]
                        .block()
                        .values.squeeze(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    ]
            else:
                energy_tensor = energies.block().values
                energy_tensor.backward(torch.ones_like(energy_tensor))
                results_as_numpy_arrays["forces"] = [
                    -system.positions.grad.cpu().numpy() for system in systems
                ]
                if all(atoms.pbc.all() for atoms in atoms_list):
                    results_as_numpy_arrays["stress"] = [
                        strain.grad.cpu().numpy() / atoms.cell.volume
                        for strain, atoms in zip(strains, atoms_list, strict=False)
                    ]
        if was_single:
            for key, value in results_as_numpy_arrays.items():
                results_as_numpy_arrays[key] = value[0]
        return results_as_numpy_arrays

    def _ase_properties_to_metatensor_outputs(
        self,
        properties,
        *,
        calculate_forces,
        calculate_stress,
        calculate_stresses,
    ):
        energy_properties = []
        for p in properties:
            if p in ["energy", "energies", "forces", "stress", "stresses"]:
                energy_properties.append(p)
            else:
                raise PropertyNotImplementedError(
                    f"property '{p}' it not yet supported by this calculator, "
                    "even if it might be supported by the model"
                )

        metatensor_outputs = {}

        # Only add energy output if the model supports it
        if self._energy_key is not None:
            output = ModelOutput(
                quantity="energy",
                unit="ev",
                explicit_gradients=[],
            )

            if "energies" in properties or "stresses" in properties:
                output.per_atom = True
            else:
                output.per_atom = False

            metatensor_outputs[self._energy_key] = output
        if calculate_forces and self.parameters["non_conservative"]:
            metatensor_outputs[self._nc_forces_key] = ModelOutput(
                quantity="force",
                unit="eV/Angstrom",
                per_atom=True,
            )

        if calculate_stress and self.parameters["non_conservative"]:
            metatensor_outputs[self._nc_stress_key] = ModelOutput(
                quantity="pressure",
                unit="eV/Angstrom^3",
                per_atom=False,
            )

        if calculate_stresses and self.parameters["non_conservative"]:
            raise NotImplementedError(
                "non conservative, per-atom stress is not yet implemented"
            )

        available_outputs = self._model.capabilities().outputs
        for key in metatensor_outputs:
            if key not in available_outputs:
                raise ValueError(f"this model does not support '{key}' output")

        return metatensor_outputs


def _get_ase_input(
    atoms: ase.Atoms,
    name: str,
    option: ModelOutput,
    dtype: torch.dtype,
    device: torch.device,
) -> "TensorMap":
    if name not in ARRAY_QUANTITIES:
        raise ValueError(
            f"The model requested '{name}', which is not available in `ase`."
        )

    infos = ARRAY_QUANTITIES[name]

    values = infos["getter"](atoms)
    if values.shape[0] != len(atoms):
        raise NotImplementedError(
            f"The model requested the '{name}' input, "
            f"but the data is not per-atom (shape {values.shape}). "
        )
    # Shape: (n_atoms, n_components) -> (n_atoms, n_components, /* n_properties */ 1)
    # for metatensor
    values = torch.tensor(values[..., None])

    components = []
    if values.shape[1] != 1:
        components.append(Labels(["xyz"], torch.arange(values.shape[1]).reshape(-1, 1)))

    block = TensorBlock(
        values,
        samples=Labels(
            ["system", "atom"],
            torch.vstack(
                [torch.full((values.shape[0],), 0), torch.arange(values.shape[0])]
            ).T,
        ),
        components=components,
        properties=Labels([infos["quantity"]], torch.tensor([[0]])),
    )

    tensor = TensorMap(Labels(["_"], torch.tensor([[0]])), [block])

    tensor.set_info("quantity", infos["quantity"])
    tensor.set_info("unit", infos["unit"])

    tensor = tensor.to(dtype=dtype, device=device)
    return tensor


def _ase_to_torch_data(atoms, dtype, device):
    """Get the positions, cell and pbc from ASE atoms as torch tensors"""

    types = torch.from_numpy(atoms.numbers).to(dtype=torch.int32, device=device)
    positions = torch.from_numpy(atoms.positions).to(dtype=dtype).to(device=device)
    cell = torch.zeros((3, 3), dtype=dtype, device=device)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool, device=device)

    cell[pbc] = torch.tensor(atoms.cell[atoms.pbc], dtype=dtype, device=device)

    return types, positions, cell, pbc


def _full_3x3_to_voigt_6_stress(stress):
    """
    Re-implementation of ``ase.stress.full_3x3_to_voigt_6_stress`` which does not do the
    stress symmetrization correctly (they do ``(stress[1, 2] + stress[1, 2]) / 2.0``)
    """
    return np.array(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            (stress[1, 2] + stress[2, 1]) / 2.0,
            (stress[0, 2] + stress[2, 0]) / 2.0,
            (stress[0, 1] + stress[1, 0]) / 2.0,
        ]
    )
